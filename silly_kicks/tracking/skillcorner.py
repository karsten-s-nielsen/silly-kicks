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

import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

# Same-provider single-source: spadl.skillcorner (the SkillCorner EVENTS converter) owns the
# nominal period offsets; SK tracking imports the SAME constant so frames match events (kills
# duplicated-truth #3). SK P2 raw clock starts exactly at the nominal 2700 (verified). NB: this
# is NOT the metrica cross-wire the review flagged -- metrica.py has its own per-period-min clock.
from silly_kicks.spadl.skillcorner import _PERIOD_START_SECONDS

from ._gk_identification import (
    _SPADL_X_MAX,
    _SPADL_X_MIN,
    _SPADL_Y_MAX,
    _SPADL_Y_MIN,
    derive_goalkeepers,
)
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

# S1 within-pitch invariant (CR 2026-06-30, ADR-024 amendment). LAYERED:
#  * PLAYERS: the per-row CATASTROPHIC backstop is the PRE-EXISTING `derive_goalkeepers` raise at
#    [_SPADL_X_MIN, _SPADL_X_MAX] x [_SPADL_Y_MIN, _SPADL_Y_MAX] (it protects the positional GK-id
#    algorithm from garbage coords -- a sign/origin transform break trips it loudly; do NOT soften
#    it). S1 adds a THIN observability band a fixed margin INSIDE that bound (expressed RELATIVE to
#    the shared constant so the two never drift): mildly-off-pitch players warn+count before they
#    would crash. Legit behind-goal keepers (SPADL x ~= -7.5..116) stay inside the band.
#  * BALL: `derive_goalkeepers` is player-ONLY, so the ball has NO existing guard -- S1's warn+count
#    is its SOLE off-pitch signal (and correctly never crashes: a long clearance out of play is
#    legit; only the aggregate rate-gate flags a systematic ball break).
# The deferred CI rate-gate is the SYSTEMATIC backstop for both. TOL_BALL provisional (re-calibrate
# from the measured bronze on the pining corpus). The shared SPADL bounds are imported at the top.
_S1_PLAYER_MARGIN = 3.0  # S1 player band sits this far INSIDE the shared derive_goalkeepers bound
_TOL_BALL = 30.0  # ball tolerance (m); ball has no existing guard -> S1 is its sole signal


def _count_gross_off_pitch(x: pd.Series, y: pd.Series, is_ball: pd.Series) -> int:
    """Count rows whose SPADL coords fall off-pitch beyond the per-kind S1 tolerance (CR 2026-06-30).
    Pure; no mutation. Players use the shared ``derive_goalkeepers`` bound minus
    ``_S1_PLAYER_MARGIN`` (a thin band just inside the catastrophic crash); the ball uses the wider
    standalone ``_TOL_BALL`` (no existing guard)."""
    px = x.to_numpy(float)
    py = y.to_numpy(float)
    ball = is_ball.to_numpy(bool)
    px_lo, px_hi = _SPADL_X_MIN + _S1_PLAYER_MARGIN, _SPADL_X_MAX - _S1_PLAYER_MARGIN
    py_lo, py_hi = _SPADL_Y_MIN + _S1_PLAYER_MARGIN, _SPADL_Y_MAX - _S1_PLAYER_MARGIN
    x_lo = np.where(ball, -_TOL_BALL, px_lo)
    x_hi = np.where(ball, 105.0 + _TOL_BALL, px_hi)
    y_lo = np.where(ball, -_TOL_BALL, py_lo)
    y_hi = np.where(ball, 68.0 + _TOL_BALL, py_hi)
    off = (px < x_lo) | (px > x_hi) | (py < y_lo) | (py > y_hi)
    return int(np.nansum(off))


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

    # S1 within-pitch invariant (CR 2026-06-30): a correct centre-origin -> SPADL transform keeps
    # bodies within the pitch except a tolerance for legitimately off-pitch ones (keepers behind the
    # goal line; out-of-play ball). Per-row GROSS off-pitch -> warn + count; NEVER clamp/crash (one
    # noisy row must not fail a match). A SYSTEMATIC fraction off-pitch is caught by the deferred CI
    # rate-gate. TOL provisional -- re-calibrate from the measured bronze range on the pining corpus.
    n_gross_off_pitch = _count_gross_off_pitch(df["x"], df["y"], df["is_ball"])
    if n_gross_off_pitch:
        warnings.warn(
            f"skillcorner.convert_to_frames: {n_gross_off_pitch} row(s) off-pitch beyond the S1 "
            "tolerance (player band inside the derive_goalkeepers bound / ball "
            f"{_TOL_BALL} m) -- possible coordinate-transform or ingestion issue upstream. "
            "Not clamped (catastrophic player coords are separately raised by derive_goalkeepers).",
            stacklevel=2,
        )

    df = df.sort_values(["player_id", "frame_id"]).reset_index(drop=True)
    df = _derive_speed(df)

    # GK identification: TRUST the native roster is_goalkeeper (CR 2026-06-30 S1). SkillCorner ships a
    # reliable per-player GK role (skillcorner_matches.position_acronym, verified 1/team); use it as-is
    # and DERIVE ONLY for a (game, team) whose native flag is absent. The previous unconditional
    # derive_goalkeepers re-derived positionally every call -- stable on a full match but on a 250-frame
    # batch a transiently goal-parked outfielder gets flagged, and across the lakehouse's per-batch
    # builds the union reached ~15 "keepers"/team. Trusting the batch-invariant roster makes SkillCorner
    # batching-immune (and matches gradientsports.py / sportec.py, which already roster-trust).
    players = df[~df["is_ball"].astype(bool)]
    native_gk = {
        (str(g), str(t)): set(grp.loc[grp["is_goalkeeper"].astype(bool), "player_id"].dropna().astype(str))
        for (g, t), grp in players.groupby(["game_id", "team_id"], sort=False)
    }
    absent = [(g, t) for (g, t), gks in native_gk.items() if not gks]
    derived_picks: dict[tuple[str, str], list[str]] = {}
    if absent:  # fallback ONLY for teams with no native GK (a data-quality edge, not the norm)
        df, derived_picks = derive_goalkeepers(
            df, teams=pd.MultiIndex.from_tuples(absent, names=["game_id", "team_id"])
        )
    n_derived = len(derived_picks)
    df["is_goalkeeper_source"] = None
    for g, t in native_gk:
        m = (df["game_id"].astype(str) == g) & (df["team_id"].astype(str) == t) & ~df["is_ball"].astype(bool)
        df.loc[m, "is_goalkeeper_source"] = "derived" if (g, t) in derived_picks else "native"

    # S2 guard (CR 2026-06-30): a resolved per-(game, team) GK count outside [1, 2] is implausible
    # (squad-wide contamination, or a missing roster). Warn + count; never fires once the 1/team roster
    # is trusted, but guards the derive-fallback path + any future provider/path that derives.
    resolved = df[~df["is_ball"].astype(bool)]
    gk_count: dict[tuple[str, str], int] = {
        (str(g), str(t)): int(grp["player_id"].nunique())
        for (g, t), grp in resolved[resolved["is_goalkeeper"].astype(bool)].groupby(["game_id", "team_id"], sort=False)
    }
    over = {f"{g}/{t}": n for (g, t), n in gk_count.items() if n > 2}
    zero = [f"{g}/{t}" for g, t in native_gk if (g, t) not in gk_count]
    n_implausible_gk_teams = len(over) + len(zero)
    if n_implausible_gk_teams:
        warnings.warn(
            f"skillcorner.convert_to_frames: {n_implausible_gk_teams} (game, team) with implausible "
            f"GK count -- >2: {over or '{}'}; 0: {zero or '[]'}. Likely squad-wide GK contamination "
            "or a missing roster flag.",
            stacklevel=2,
        )

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
        n_gross_off_pitch=n_gross_off_pitch,
        n_implausible_gk_teams=n_implausible_gk_teams,
    )
    return final, report
