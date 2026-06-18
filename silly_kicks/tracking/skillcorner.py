"""SkillCorner bronze->frame builder (TF-23, ADR-034).

Pure ``pd.DataFrame -> (pd.DataFrame, TrackingConversionReport)`` builder consuming the
post-join SkillCorner bronze (``bronze.skillcorner_tracking`` joined with
``bronze.skillcorner_matches`` for team/GK), parallel to ``tracking.sportec``. Single-
sources the coordinate (centre-origin -> SPADL 105x68), ``ball_z`` recovery, period-
relative clock, id-namespacing, GK derivation, speed, and geometric LTR orientation
that the luxury-lakehouse previously duplicated. See spec
``2026-06-18-tf23-skillcorner-metrica-bronze-frame-builders-design.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

# Same-provider single-source: spadl.skillcorner (the SkillCorner EVENTS converter) owns the
# nominal period offsets; SK tracking imports the SAME constant so frames match events (kills
# duplicated-truth #3). SK P2 raw clock starts exactly at the nominal 2700 (verified). NB: this
# is NOT the metrica cross-wire the review flagged -- metrica.py has its own per-period-min clock.
from silly_kicks.spadl.skillcorner import _PERIOD_START_SECONDS

from ._gk_identification import derive_goalkeepers
from .direction import orient_frames_to_ltr_by_geometry, require_et_direction
from .schema import SKILLCORNER_TRACKING_FRAMES_COLUMNS, TrackingConversionReport
from .utils import _derive_speed, orient_frames_to_ltr

if TYPE_CHECKING:
    from .preprocess import PreprocessConfig

EXPECTED_INPUT_COLUMNS: tuple[str, ...] = (
    "match_id",
    "period",
    "frame",
    "timestamp",
    "player_id",
    "team_id",
    "is_goalkeeper",
    "x",
    "y",
    "ball_x",
    "ball_y",
    "ball_z",
    "is_visible",
    "frame_rate",
)


def convert_to_frames(
    bronze: pd.DataFrame,
    *,
    home_team_id: Any,
    output_convention: Literal["absolute_frame", "ltr"] = "ltr",
    home_team_start_left: bool | None = None,
    home_team_start_left_extratime: bool | None = None,
    preprocess: PreprocessConfig | None = None,
) -> tuple[pd.DataFrame, TrackingConversionReport]:
    """Convert post-join SkillCorner bronze tracking to canonical SPADL frames.

    Parameters
    ----------
    bronze : pd.DataFrame
        SkillCorner tracking bronze joined with ``skillcorner_matches`` (team/GK).
        Required columns: see ``EXPECTED_INPUT_COLUMNS``. ``x``/``y`` are centre-origin
        meters; ``ball_z`` is real ball height; ``timestamp`` is the continuous
        broadcast clock; ``is_goalkeeper`` is the native (roster) flag.
    home_team_id : Any
        Home team id (matches ``team_id`` after stringification).
    output_convention : {"absolute_frame", "ltr"}, default "ltr"
        ``"ltr"`` orients via the geometric net (``home_team_start_left=None``) or the
        flag-based ``orient_frames_to_ltr`` (when a flag is supplied). ``"absolute_frame"``
        leaves frames unoriented (``team_attacking_direction=None``).
    home_team_start_left, home_team_start_left_extratime : bool | None
        Optional authoritative orientation flags; ``None`` => geometric orientation.
    preprocess : PreprocessConfig | None
        Optional smoothing/velocity; off by default.

    Returns
    -------
    tuple[pd.DataFrame, TrackingConversionReport]

    Examples
    --------
    Build LTR frames from bronze::

        from silly_kicks.tracking import skillcorner
        frames, report = skillcorner.convert_to_frames(bronze_df, home_team_id="31")
    """
    missing = [c for c in EXPECTED_INPUT_COLUMNS if c not in bronze.columns]
    if missing:
        raise ValueError(f"skillcorner.convert_to_frames: bronze missing column(s): {missing}")
    if home_team_start_left is not None and output_convention == "ltr":
        require_et_direction(bronze["period"], home_team_start_left_extratime, source="skillcorner convert_to_frames")

    src = bronze.copy()
    game_id = str(src["match_id"].iloc[0])
    frame_rate = float(src["frame_rate"].iloc[0]) if "frame_rate" in src else 10.0

    # --- player rows ---
    players = src[
        ["frame", "period", "timestamp", "player_id", "team_id", "is_goalkeeper", "x", "y", "is_visible"]
    ].copy()
    players = players.rename(columns={"frame": "frame_id", "period": "period_id", "timestamp": "time_seconds"})
    players["x"] = players["x"] + 52.5
    players["y"] = players["y"] + 34.0
    players["z"] = np.nan
    players["visibility"] = players.pop("is_visible")
    players["is_ball"] = False
    players["player_id"] = players["player_id"].astype(str)
    players["team_id"] = players["team_id"].astype(str)

    # --- ball rows (one per (frame, period); recover ball_z) ---
    ball = (
        src[["frame", "period", "timestamp", "ball_x", "ball_y", "ball_z"]]
        .drop_duplicates(subset=["frame", "period"])
        .copy()
    )
    ball = ball.rename(
        columns={
            "frame": "frame_id",
            "period": "period_id",
            "timestamp": "time_seconds",
            "ball_x": "x",
            "ball_y": "y",
            "ball_z": "z",
        }
    )
    ball["x"] = ball["x"] + 52.5
    ball["y"] = ball["y"] + 34.0
    ball["player_id"] = None
    ball["team_id"] = None
    ball["is_goalkeeper"] = False
    ball["visibility"] = None
    ball["is_ball"] = True

    df = pd.concat([players, ball], ignore_index=True)
    df["game_id"] = game_id
    df["frame_rate"] = frame_rate
    df["source_provider"] = "skillcorner"
    df["ball_state"] = None
    df["team_attacking_direction"] = None
    df["confidence"] = None
    df["speed"] = np.nan
    df["speed_source"] = None

    # period-relative clock via the SINGLE-SOURCED nominal constant (matches the SK events
    # converter, so action<->frame linkage is exact). SkillCorner's `timestamp` is
    # nominal-aligned PER PERIOD -- each period's clock starts at its nominal boundary
    # regardless of prior stoppage (verified: P2 `timestamp` = exactly 2700.0 in all 10
    # pining matches, despite P1 ending at 2778-3023 s), so `timestamp - nominal` is exactly
    # period-relative-from-0 and matches the events. NOT `period_boundaries` (frame indices,
    # not a seconds-clock; O3 closed). (Metrica differs: per-(period)-min rebase -- mixed raw
    # clocks.)
    df["time_seconds"] = df["time_seconds"] - df["period_id"].map(_PERIOD_START_SECONDS).fillna(0.0).astype(float)

    df = df.sort_values(["player_id", "frame_id"]).reset_index(drop=True)
    df = _derive_speed(df)

    # GK derivation + agreement-based source (mirror the kloppy gateway)
    native_gk = {
        (str(g), str(t)): set(grp.loc[grp["is_goalkeeper"], "player_id"].dropna().astype(str))
        for (g, t), grp in df[~df["is_ball"]].groupby(["game_id", "team_id"], sort=False)
    }
    df, derived_picks = derive_goalkeepers(df)
    n_derived = 0
    df["is_goalkeeper_source"] = None
    for (g, t), algo in derived_picks.items():
        source_val = "native" if set(algo) == native_gk.get((g, t), set()) else "derived"
        n_derived += source_val == "derived"
        m = (df["game_id"] == g) & (df["team_id"] == t) & ~df["is_ball"]
        df.loc[m, "is_goalkeeper_source"] = source_val

    final = pd.DataFrame({c: df[c] for c in SKILLCORNER_TRACKING_FRAMES_COLUMNS})
    for c, dt in SKILLCORNER_TRACKING_FRAMES_COLUMNS.items():
        if dt == "bool":
            final[c] = final[c].astype("bool")
        elif dt in {"int64", "float64"}:
            final[c] = pd.to_numeric(final[c], errors="coerce").astype(dt)  # type: ignore[arg-type]
        else:
            final[c] = final[c].astype(object)

    if output_convention == "ltr":
        if home_team_start_left is None:
            final = orient_frames_to_ltr_by_geometry(
                final, home_team_id=str(home_team_id), source="skillcorner", game_id=game_id
            )
        else:
            final = orient_frames_to_ltr(
                final,
                home_team_id=str(home_team_id),
                home_team_start_left=home_team_start_left,
                home_team_start_left_extratime=home_team_start_left_extratime,
            )

    if preprocess is not None:
        from .preprocess import derive_velocities, interpolate_frames, smooth_frames
        from .preprocess._resolve import resolve_preprocess

        cfg = resolve_preprocess(preprocess, provider="skillcorner")
        if cfg.interpolation_method is not None:
            final = interpolate_frames(final, config=cfg)
        if cfg.smoothing_method is not None:
            final = smooth_frames(final, config=cfg)
        if cfg.derive_velocity:
            final = derive_velocities(final, config=cfg)

    report = TrackingConversionReport(
        provider="skillcorner",
        total_input_frames=int(src[["frame", "period"]].drop_duplicates().shape[0]),
        total_output_rows=len(final),
        n_periods=int(final["period_id"].nunique()),
        frame_coverage_per_period={int(p): 1.0 for p in final["period_id"].unique()},
        ball_out_seconds_per_period={},
        nan_rate_per_column={c: float(final[c].isna().mean()) for c in final.columns},
        derived_speed_rows=int((final["speed_source"] == "derived").sum()),
        unrecognized_player_ids=set(),
        n_teams_gk_derived=n_derived,
        derived_gk_picks=derived_picks,
    )
    return final, report
