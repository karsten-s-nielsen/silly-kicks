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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

# Same-provider single-source: spadl.skillcorner (the SkillCorner EVENTS converter) owns the
# nominal period offsets; SK tracking imports the SAME constant so frames match events (kills
# duplicated-truth #3). SK P2 raw clock starts exactly at the nominal 2700 (verified). NB: this
# is NOT the metrica cross-wire the review flagged -- metrica.py has its own per-period-min clock.
from silly_kicks.spadl.skillcorner import _PERIOD_START_SECONDS, _scale_to_spadl

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
    "pitch_length",
    "pitch_width",
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
# Calibrated 2026-07-14 on the 10 PUBLIC pining matches (10.0M rows, correct transform):
# the largest real ball excursion is 9.00 m. The previous 30.0 m sat above every real value,
# so the ball tolerance could never fire. 15.0 m keeps 67% headroom over the worst observed
# and zero public rows exceed it. (Calibrating on the 98 private matches would be circular --
# they are the data under validation.) See spec 4.4.
_TOL_BALL = 15.0  # ball tolerance (m); ball has no existing guard -> S1 is its sole signal


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


# Per-match rate-gate thresholds (spec 4.4), calibrated on the public 10:
#   worst clean player_frac(>3 m) = 0.00086  ->  0.005 leaves a 5.8x margin
#   worst clean ball_frac(>10 m)  = 0.00000  ->  0.0005 is the noise floor
# A catastrophic sign/origin break measures 0.34139 -- it exceeds the player threshold by 68x.
# A 4 m PITCH-DIMENSION error measures 0.00095 and does NOT trip: this gate cannot see one, and
# neither can action-frame co-location (events and tracking read the same metadata and move
# together). That limitation is deliberate, documented, and pinned by a test.
_PLAYER_OFF_PITCH_RATE_MAX = 0.005
_BALL_OFF_PITCH_RATE_MAX = 0.0005
_PLAYER_RATE_TOL = 3.0
_BALL_RATE_TOL = 10.0


@dataclass(frozen=True)
class GeometryGateReport:
    """Outcome of the per-match geometry admission gate (spec 4.4)."""

    excluded: bool
    reason: str
    player_off_pitch_rate: float
    ball_off_pitch_rate: float


def geometry_rate_gate(frames: pd.DataFrame) -> GeometryGateReport:
    """Per-match geometry admission (spec 4.4). Pure; no I/O, no mutation.

    EXCLUDES a match whose off-pitch RATE exceeds the public-10-calibrated thresholds. This is
    the systematic backstop the S1 comment called 'deferred' -- the per-row invariant only warns,
    which is invisible in a batch log.
    """
    x = frames["x"].to_numpy(float)
    y = frames["y"].to_numpy(float)
    is_ball = frames["is_ball"].to_numpy(bool)
    exc = np.maximum(
        np.maximum(np.maximum(-x, x - 105.0), 0.0),
        np.maximum(np.maximum(-y, y - 68.0), 0.0),
    )
    players, balls = exc[~is_ball], exc[is_ball]
    p_rate = float((players > _PLAYER_RATE_TOL).mean()) if len(players) else 0.0
    b_rate = float((balls > _BALL_RATE_TOL).mean()) if len(balls) else 0.0
    reasons = []
    if p_rate > _PLAYER_OFF_PITCH_RATE_MAX:
        reasons.append(f"player off-pitch rate {p_rate:.5f} > {_PLAYER_OFF_PITCH_RATE_MAX}")
    if b_rate > _BALL_OFF_PITCH_RATE_MAX:
        reasons.append(f"ball off-pitch rate {b_rate:.5f} > {_BALL_OFF_PITCH_RATE_MAX}")
    return GeometryGateReport(
        excluded=bool(reasons),
        reason="; ".join(reasons),
        player_off_pitch_rate=p_rate,
        ball_off_pitch_rate=b_rate,
    )


def convert_to_frames(
    bronze: pd.DataFrame,
    *,
    home_team_id: Any,
    output_convention: Literal["absolute_frame", "ltr"] = "ltr",
    home_team_start_left: bool | None = None,
    home_team_start_left_extratime: bool | None = None,
    preprocess: PreprocessConfig | None = None,
    assume_standard_pitch: bool = False,
) -> tuple[pd.DataFrame, TrackingConversionReport]:
    """Convert post-join SkillCorner bronze tracking to canonical SPADL frames.

    Parameters
    ----------
    bronze : pd.DataFrame
        SkillCorner tracking bronze joined with ``skillcorner_matches`` (team/GK).
        Required columns: see ``EXPECTED_INPUT_COLUMNS``. ``x``/``y`` are centre-origin
        meters; ``ball_z`` is real ball height; ``timestamp`` is the continuous
        broadcast clock; ``is_goalkeeper`` is the native (roster) flag.
        ``pitch_length``/``pitch_width`` are the match pitch dimensions (metres) used
        to SCALE centre-origin metres to the SPADL 105x68 frame -- a fixed offset
        would land the goal line ~2 m off on a non-105 m pitch (spec 3.4). They are
        REQUIRED unless ``assume_standard_pitch=True``.
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
    assume_standard_pitch : bool, default False
        Explicit opt-in to a 105x68 pitch when ``pitch_length``/``pitch_width`` are
        absent from ``bronze``. Fail-closed by default (spec 3.4 / reviewer m1): a
        silent 105x68 default would reproduce the very goal-line defect this fixes, and
        a warning is invisible in a DGX batch log -- so missing dims RAISE unless this
        is set. Pass ``True`` only when the pitch is known to be 105x68.

    Returns
    -------
    tuple[pd.DataFrame, TrackingConversionReport]

    Examples
    --------
    Build LTR frames from bronze::

        from silly_kicks.tracking import skillcorner
        frames, report = skillcorner.convert_to_frames(bronze_df, home_team_id="31")
    """
    # Pitch dims are validated separately so an absent dim raises the INFORMATIVE fail-closed
    # message below (a generic "missing column" would hide WHY it matters).
    required = [c for c in EXPECTED_INPUT_COLUMNS if not c.startswith("pitch_")]
    missing = [c for c in required if c not in bronze.columns]
    if missing:
        raise ValueError(f"skillcorner.convert_to_frames: bronze missing column(s): {missing}")
    if assume_standard_pitch:
        pitch_length, pitch_width = 105.0, 68.0
    else:
        missing_dims = [c for c in ("pitch_length", "pitch_width") if c not in bronze.columns]
        if missing_dims:
            # Fail-closed (spec 3.4 / reviewer m1): defaulting to 105x68 silently reproduces the
            # goal-line defect this fixes, and a warning is invisible in a DGX batch log.
            raise ValueError(
                f"skillcorner.convert_to_frames: bronze missing {missing_dims}. Pitch dimensions are "
                "REQUIRED -- defaulting to 105x68 silently reproduces the goal-line defect this "
                "fixes (spec 3.4). Pass assume_standard_pitch=True only if you know the pitch is "
                "105x68."
            )
        pitch_length = float(bronze["pitch_length"].iloc[0])
        pitch_width = float(bronze["pitch_width"].iloc[0])
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
    # SCALE centre-origin metres -> SPADL 105x68 (NOT a fixed +52.5/+34 offset: on a non-105 m pitch
    # that lands the goal line ~2 m off, spec 3.4). _scale_to_spadl is the events converter's affine
    # map WITHOUT its clamp -- tracking is full of legitimately off-pitch positions (a keeper behind
    # his line, a ball across the goal line, which is what a goal IS); clamping would erase goal-vs-save.
    players["x"], players["y"] = _scale_to_spadl(players["x"], players["y"], pitch_length, pitch_width)
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
    # NEVER clamp: an off-pitch ball across the goal line is what a goal IS (spec 3.4).
    ball["x"], ball["y"] = _scale_to_spadl(ball["x"], ball["y"], pitch_length, pitch_width)
    ball["player_id"] = None
    ball["team_id"] = None
    ball["is_goalkeeper"] = False
    ball["visibility"] = None
    ball["is_ball"] = True

    df = pd.concat([players, ball], ignore_index=True)
    df["game_id"] = game_id
    df["frame_rate"] = frame_rate
    df["source_provider"] = "skillcorner"
    # SkillCorner's native feed carries no reliable dead-ball signal, so default to "alive"
    # (in-play). None is NOT a valid ball_state (schema value-set is {"alive","dead"}) and makes
    # the strict `== "alive"` domain filter in xS/xCross drop EVERY SkillCorner frame -> 0 training
    # rows (the kloppy gateway set real states, so this only regressed once the loader rerouted to
    # this native builder). "alive" makes that filter a no-op for SkillCorner (use all frames);
    # `== "dead"` / not-dead consumers are unchanged (None and "alive" are both non-dead).
    df["ball_state"] = "alive"
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

    # S1 SYSTEMATIC rate-gate (spec 4.4): the per-row _count_gross_off_pitch invariant above only
    # WARNS -- a warning is invisible in a DGX batch log. The gate turns a SYSTEMATIC off-pitch
    # fraction (a sign/origin transform break) into a machine-observable admission decision on the
    # report; the loader reads report.geometry_excluded to DROP the match. Runs on the final,
    # oriented frames -- a point-reflection LTR flip is an isometry, so off-pitch magnitude is
    # orientation-invariant. Report is frozen, so the fields ride the constructor (not a mutation).
    gate = geometry_rate_gate(final)
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
        geometry_excluded=gate.excluded,
        geometry_reason=gate.reason,
        player_off_pitch_rate=gate.player_off_pitch_rate,
        ball_off_pitch_rate=gate.ball_off_pitch_rate,
    )
    return final, report
