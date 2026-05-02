"""Kloppy TrackingDataset gateway for silly_kicks.tracking.

Covers Metrica + SkillCorner via kloppy 3.18+ tracking parsers. Sportec
and PFF intentionally raise NotImplementedError --- route through their
native adapters (silly_kicks.tracking.sportec / silly_kicks.tracking.pff)
for symmetry with silly_kicks.spadl.pff (PR-S18) and failure isolation.

See ADR-004 (silly-kicks 2.7.0) for the architectural rationale.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from kloppy.domain import (  # type: ignore[reportMissingImports]
    Dimension,
    MetricPitchDimensions,
    Orientation,
    Provider,
    TrackingDataset,
)

from .schema import KLOPPY_TRACKING_FRAMES_COLUMNS, TrackingConversionReport
from .sportec import _resolve_output_convention
from .utils import _derive_speed

_PROVIDER_NAME_MAP: dict[Provider, str] = {
    Provider.METRICA: "metrica",
    Provider.SKILLCORNER: "skillcorner",
}


def convert_to_frames(
    dataset: TrackingDataset,
    preserve_native: list[str] | None = None,
    *,
    output_convention: Literal["absolute_frame", "ltr"] | None = None,
) -> tuple[pd.DataFrame, TrackingConversionReport]:
    """Convert a kloppy TrackingDataset to canonical KLOPPY_TRACKING_FRAMES_COLUMNS schema.

    Dispatches on ``dataset.metadata.provider``. ``Provider.PFF`` and
    ``Provider.SPORTEC`` raise ``NotImplementedError`` --- route through
    ``silly_kicks.tracking.pff`` and ``silly_kicks.tracking.sportec``.

    Parameters
    ----------
    dataset : kloppy.domain.TrackingDataset
        Output of e.g. ``kloppy.metrica.load_tracking_csv`` or
        ``kloppy.skillcorner.load_tracking``.
    preserve_native : list[str] | None
        Reserved for future PR --- pass through optional input columns.
    output_convention : {"absolute_frame", "ltr"} | None, default None
        Coordinate convention of the returned frames. ``"absolute_frame"``
        (the historical default) emits frames in absolute-frame-home-right
        convention with per-row ``team_attacking_direction``. ``"ltr"`` applies
        :func:`silly_kicks.tracking.utils.play_left_to_right` internally so the
        output is in canonical SPADL "all teams attack left-to-right". Passing
        ``None`` emits a ``DeprecationWarning`` and defaults to
        ``"absolute_frame"`` -- callers should pick one explicitly. See
        ADR-006 (silly-kicks 3.0.0).

    Returns
    -------
    frames : pd.DataFrame
        KLOPPY_TRACKING_FRAMES_COLUMNS-shaped output, in the convention
        requested by ``output_convention``.
    report : TrackingConversionReport

    Examples
    --------
    Load a Metrica match via kloppy and convert in absolute frame::

        import kloppy
        from silly_kicks.tracking import kloppy as tracking_kloppy
        ds = kloppy.metrica.load_tracking_csv(home="home.csv", away="away.csv")
        frames, report = tracking_kloppy.convert_to_frames(
            ds, output_convention="absolute_frame",
        )
    """
    _ = preserve_native  # reserved
    output_convention = _resolve_output_convention(output_convention, _adapter_name="kloppy")
    provider = dataset.metadata.provider
    if provider == Provider.PFF:
        raise NotImplementedError(
            "PFF tracking via kloppy is supported but disabled in PR-S19; "
            "route through silly_kicks.tracking.pff for symmetry with "
            "silly_kicks.spadl.pff (ADR-004)."
        )
    if provider == Provider.SPORTEC:
        raise NotImplementedError(
            "Sportec tracking has no kloppy parser; route through silly_kicks.tracking.sportec (ADR-004)."
        )
    provider_name = _PROVIDER_NAME_MAP.get(provider)
    if provider_name is None:
        raise NotImplementedError(f"Provider {provider} not supported in PR-S19")

    transformed = dataset.transform(
        to_pitch_dimensions=MetricPitchDimensions(
            x_dim=Dimension(0, 105.0),
            y_dim=Dimension(0, 68.0),
            standardized=False,
            pitch_length=105.0,
            pitch_width=68.0,
        ),
        to_orientation=Orientation.HOME_AWAY,
    )

    home_team = transformed.metadata.teams[0]
    home_team_id = str(home_team.team_id)

    rows: list[dict] = []
    frame_rate = float(transformed.metadata.frame_rate or 25.0)
    game_id_value = str(transformed.metadata.game_id) if transformed.metadata.game_id is not None else "synthetic"

    for frame in transformed.records:
        period_id = int(frame.period.id)
        time_seconds = float(frame.timestamp.total_seconds())
        ball_state_str = str(frame.ball_state.value).lower() if frame.ball_state is not None else None
        for player, pdata in frame.players_data.items():
            if pdata.coordinates is None:
                continue
            team_id_str = str(player.team.team_id)
            is_home = team_id_str == home_team_id
            is_gk = player.starting_position is not None and "Goalkeeper" in str(player.starting_position)
            rows.append(
                {
                    "game_id": game_id_value,
                    "period_id": period_id,
                    "frame_id": frame.frame_id,
                    "time_seconds": time_seconds,
                    "frame_rate": frame_rate,
                    "player_id": str(player.player_id),
                    "team_id": team_id_str,
                    "is_ball": False,
                    "is_goalkeeper": is_gk,
                    "x": pdata.coordinates.x,
                    "y": pdata.coordinates.y,
                    "z": float("nan"),
                    "speed": pdata.speed if pdata.speed is not None else float("nan"),
                    "speed_source": "native" if pdata.speed is not None else None,
                    "ball_state": ball_state_str,
                    "team_attacking_direction": "ltr" if is_home else "rtl",
                    "confidence": None,
                    "visibility": None,
                    "source_provider": provider_name,
                }
            )
        if frame.ball_coordinates is not None:
            ball_z_raw = getattr(frame.ball_coordinates, "z", None)
            ball_z = float(ball_z_raw) if ball_z_raw is not None else float("nan")
            rows.append(
                {
                    "game_id": game_id_value,
                    "period_id": period_id,
                    "frame_id": frame.frame_id,
                    "time_seconds": time_seconds,
                    "frame_rate": frame_rate,
                    "player_id": None,
                    "team_id": None,
                    "is_ball": True,
                    "is_goalkeeper": False,
                    "x": frame.ball_coordinates.x,
                    "y": frame.ball_coordinates.y,
                    "z": ball_z,
                    "speed": frame.ball_speed if frame.ball_speed is not None else float("nan"),
                    "speed_source": "native" if frame.ball_speed is not None else None,
                    "ball_state": ball_state_str,
                    "team_attacking_direction": None,
                    "confidence": None,
                    "visibility": None,
                    "source_provider": provider_name,
                }
            )

    df = pd.DataFrame(rows)
    if df["speed"].isna().any():
        df = _derive_speed(df)

    final = pd.DataFrame({col: df[col] for col in KLOPPY_TRACKING_FRAMES_COLUMNS})
    for col, dtype_str in KLOPPY_TRACKING_FRAMES_COLUMNS.items():
        if dtype_str == "bool":
            final[col] = final[col].astype("bool")
        elif dtype_str in {"int64", "float64"}:
            final[col] = pd.to_numeric(final[col], errors="coerce").astype(dtype_str)  # type: ignore[arg-type]
        elif dtype_str == "object":
            final[col] = final[col].astype(object)

    n_input_frames = len(transformed.records)
    n_periods = len({f.period.id for f in transformed.records})
    cov: dict[int, float] = {}
    ball_out: dict[int, float] = {}
    for p, g in final.groupby("period_id", sort=True):
        ball_g = g[g["is_ball"]]
        cov[int(p)] = 1.0  # type: ignore[arg-type]
        if len(ball_g):
            dt = 1.0 / float(ball_g["frame_rate"].iloc[0])
            ball_out[int(p)] = float((ball_g["ball_state"] == "dead").sum() * dt)  # type: ignore[arg-type]

    nan_rate = {col: float(final[col].isna().mean()) for col in final.columns}

    report = TrackingConversionReport(
        provider=provider_name,
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

    return final, report
