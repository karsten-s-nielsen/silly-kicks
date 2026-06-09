"""Kloppy TrackingDataset gateway for silly_kicks.tracking.

Covers Metrica + SkillCorner via kloppy 3.18+ tracking parsers. Sportec
and Gradient Sports intentionally raise NotImplementedError --- route through
their native adapters (silly_kicks.tracking.sportec /
silly_kicks.tracking.gradientsports) for symmetry with
silly_kicks.spadl.gradientsports (PR-S18) and failure isolation.

See ADR-004 (silly-kicks 2.7.0) for the architectural rationale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd
from kloppy.domain import (  # type: ignore[reportMissingImports]
    Dimension,
    MetricPitchDimensions,
    Orientation,
    Provider,
    TrackingDataset,
)

from ._id_compat import same_id
from .schema import KLOPPY_TRACKING_FRAMES_COLUMNS, TrackingConversionReport
from .sportec import _resolve_output_convention
from .utils import _derive_speed

if TYPE_CHECKING:
    from .preprocess import PreprocessConfig

_PROVIDER_NAME_MAP: dict[Provider, str] = {
    Provider.METRICA: "metrica",
    Provider.SKILLCORNER: "skillcorner",
}


def convert_to_frames(
    dataset: TrackingDataset,
    preserve_native: list[str] | None = None,
    *,
    output_convention: Literal["absolute_frame", "ltr"] | None = None,
    preprocess: PreprocessConfig | None = None,
) -> tuple[pd.DataFrame, TrackingConversionReport]:
    """Convert a kloppy TrackingDataset to canonical KLOPPY_TRACKING_FRAMES_COLUMNS schema.

    Dispatches on ``dataset.metadata.provider``. ``Provider.PFF`` and
    ``Provider.SPORTEC`` raise ``NotImplementedError`` --- route through
    ``silly_kicks.tracking.gradientsports`` and ``silly_kicks.tracking.sportec``.

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
            "Gradient Sports (PFF) tracking via kloppy is supported but disabled in PR-S19; "
            "route through silly_kicks.tracking.gradientsports for symmetry with "
            "silly_kicks.spadl.gradientsports (ADR-004)."
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
            # ADR-019 (4.21.1): route the orientation seam through the id-compat helper for
            # consistency with the gradientsports/sportec adapters. Both sides are already
            # `str(...)` here (no caller-dtype boundary), so this is behavior-identical -- the
            # same_id fast path (both object) adds negligible per-player overhead.
            is_home = same_id(team_id_str, home_team_id)
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

    # --- GK identification: always run B+ algorithm, agreement-based source ---
    from ._gk_identification import derive_goalkeepers

    # Capture kloppy's native picks per (game_id, team_id) before overwriting
    native_gk_picks: dict[tuple[str, str], set[str]] = {}
    player_mask = ~df["is_ball"]
    for (gid, tid), grp in df[player_mask].groupby(["game_id", "team_id"], sort=False):
        native_gks = set(grp.loc[grp["is_goalkeeper"], "player_id"].dropna().astype(str))
        native_gk_picks[(str(gid), str(tid))] = native_gks

    # Run algorithm (overwrites is_goalkeeper column)
    df, derived_picks = derive_goalkeepers(df)

    # Compute agreement-based is_goalkeeper_source per (game_id, team_id)
    # native iff algorithm picks == kloppy native picks
    n_teams_gk_derived = 0
    df["is_goalkeeper_source"] = None  # default for ball/no-team rows
    for (gid, tid), algo_picks in derived_picks.items():
        algo_set = set(algo_picks)
        native_set = native_gk_picks.get((gid, tid), set())
        source = "native" if algo_set == native_set else "derived"
        if source == "derived":
            n_teams_gk_derived += 1
        team_mask = (df["game_id"] == gid) & (df["team_id"] == tid) & ~df["is_ball"]
        df.loc[team_mask, "is_goalkeeper_source"] = source

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
        n_teams_gk_derived=n_teams_gk_derived,
        derived_gk_picks=derived_picks,
    )

    if output_convention == "ltr":
        from .utils import play_left_to_right

        final = play_left_to_right(final, home_team_id)

    if preprocess is not None:
        from .preprocess import derive_velocities, interpolate_frames, smooth_frames
        from .preprocess._resolve import resolve_preprocess

        provider_name = _PROVIDER_NAME_MAP.get(dataset.metadata.provider, str(dataset.metadata.provider).lower())
        cfg = resolve_preprocess(preprocess, provider=provider_name)
        if cfg.interpolation_method is not None:
            final = interpolate_frames(final, config=cfg)
        if cfg.smoothing_method is not None:
            final = smooth_frames(final, config=cfg)
        if cfg.derive_velocity:
            final = derive_velocities(final, config=cfg)

    return final, report
