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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from silly_kicks.id_compat import ids_match

# Same-provider single-source: spadl.skillcorner (the SkillCorner EVENTS converter) owns the
# nominal period offsets; SK tracking imports the SAME constant so frames match events (kills
# duplicated-truth #3). SK P2 raw clock starts exactly at the nominal 2700 (verified). NB: this
# is NOT the metrica cross-wire the review flagged -- metrica.py has its own per-period-min clock.
from silly_kicks.spadl.skillcorner import _PERIOD_START_SECONDS, _scale_to_spadl

from ._camera_footprint import (
    FOOTPRINT_X_COLUMNS,
    FOOTPRINT_Y_COLUMNS,
    footprint_degenerate_mask,
    in_camera_view_mask,
    orient_camera_footprint,
)
from ._gk_identification import (
    _SPADL_X_MAX,
    _SPADL_X_MIN,
    _SPADL_Y_MAX,
    _SPADL_Y_MIN,
    derive_goalkeepers,
)
from .direction import (
    _decide_ltr_flips,
    _orient_frames_to_ltr_by_geometry,
    home_attacks_right_per_period,
    require_et_direction,
)
from .schema import (
    FRAME_FLIPS_ATTR,
    PERIOD_FRAME_RANGES_ATTR,
    SKILLCORNER_CAMERA_FOOTPRINT_COLUMNS,
    SKILLCORNER_TRACKING_FRAMES_COLUMNS,
    Z_PROVENANCE,
    TrackingConversionReport,
)
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


def _resolve_pitch_dims(bronze: pd.DataFrame, assume_standard_pitch: bool) -> tuple[float, float]:
    """The fail-closed pitch-dimension rule (spec 3.4), shared by ``convert_to_frames`` and
    ``convert_camera_footprint`` so the footprint scales on the SAME dims as the frames (CL-15)."""
    if assume_standard_pitch:
        return 105.0, 68.0
    missing_dims = [c for c in ("pitch_length", "pitch_width") if c not in bronze.columns]
    if missing_dims:
        raise ValueError(
            f"skillcorner: bronze missing {missing_dims}. Pitch dimensions are REQUIRED -- defaulting to "
            "105x68 silently reproduces the goal-line defect this fixes (spec 3.4). Pass "
            "assume_standard_pitch=True only if you know the pitch is 105x68."
        )
    return float(bronze["pitch_length"].iloc[0]), float(bronze["pitch_width"].iloc[0])


def _scale_footprint(bronze: pd.DataFrame, *, pitch_length: float, pitch_width: float) -> pd.DataFrame:
    """One row per (game_id, period_id, frame_id) with the four footprint corners scaled to the SPADL frame
    via the SAME ``_scale_to_spadl`` affine as the tracking (spec section 17.7). Absent corner columns -> NaN
    (a bronze that carries no footprint; the per-profile fallback)."""
    cam_cols = [*FOOTPRINT_X_COLUMNS, *FOOTPRINT_Y_COLUMNS]
    has = all(c in bronze.columns for c in cam_cols)
    base = bronze[["match_id", "period", "frame", *(cam_cols if has else ())]].drop_duplicates(
        subset=["period", "frame"]
    )
    fp = base.rename(columns={"match_id": "game_id", "period": "period_id", "frame": "frame_id"}).reset_index(drop=True)
    if has:
        for cx, cy in zip(FOOTPRINT_X_COLUMNS, FOOTPRINT_Y_COLUMNS, strict=True):
            sx, sy = _scale_to_spadl(fp[cx].astype("float64"), fp[cy].astype("float64"), pitch_length, pitch_width)
            fp[cx] = sx.to_numpy()
            fp[cy] = sy.to_numpy()
    else:
        for c in cam_cols:
            fp[c] = np.nan
    return fp


def convert_to_frames(
    bronze: pd.DataFrame,
    *,
    home_team_id: Any,
    output_convention: Literal["absolute_frame", "ltr"] = "ltr",
    home_team_start_left: bool | None = None,
    home_team_start_left_extratime: bool | None = None,
    preprocess: PreprocessConfig | None = None,
    assume_standard_pitch: bool = False,
    period_frame_ranges: Mapping[int, tuple[int, int]] | None = None,
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
    period_frame_ranges : Mapping[int, tuple[int, int]] | None
        The PROVIDER period range of this match, ``{period_id: (first_frame, last_frame)}`` over every raw frame
        the provider labels with that period (empty frames included): :func:`provider_period_frame_ranges` of the
        raw feed. The bronze carries only frames with objects, so only the raw reader knows it. Recorded on
        ``report.provider_period_frame_ranges`` and stamped on ``frames.attrs`` (spec section 17.7, amended
        2026-10-25). RAISES ``ValueError`` when a retained frame lies outside its period's range or its period has
        none (a range from another match), ``TypeError`` on a ``(game_id, period_id)``-keyed mapping. ``None``
        records nothing.

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
    # Fail-closed dims (spec 3.4 / reviewer m1): defaulting to 105x68 silently reproduces the goal-line defect.
    pitch_length, pitch_width = _resolve_pitch_dims(bronze, assume_standard_pitch)
    if home_team_start_left is not None and output_convention == "ltr":
        require_et_direction(bronze["period"], home_team_start_left_extratime, source="skillcorner convert_to_frames")

    src = bronze.copy()
    game_id = str(src["match_id"].iloc[0])
    frame_rate = float(src["frame_rate"].iloc[0]) if "frame_rate" in src else 10.0
    provider_ranges = _check_period_frame_ranges(period_frame_ranges, src)

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
    _ball_cols = ["frame", "period", "timestamp", "ball_x", "ball_y", "ball_z"]
    _has_ball_vis = "ball_is_visible" in src.columns
    if _has_ball_vis:
        _ball_cols = [*_ball_cols, "ball_is_visible"]
    ball = src[_ball_cols].drop_duplicates(subset=["frame", "period"]).copy()
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
    # Ball-row visibility = the source `ball_data.is_detected` (spec section 17.7); None when the bronze
    # carries no detection flag (backward compatible with the pre-adapter bronze).
    ball["visibility"] = ball.pop("ball_is_visible") if _has_ball_vis else None
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
    df["speed"] = np.nan
    df["speed_source"] = None

    # --- TF-46 Arm 1 Part 2 adapter fields (spec section 17.7) ---
    # Frame-level possession carried onto every row (players + ball); absent in the bronze -> <NA>.
    _poss_cols = ["sc_possession_player_id", "sc_possession_group"]
    if all(c in src.columns for c in _poss_cols):
        poss_lu = src.drop_duplicates(subset=["frame", "period"])[["frame", "period", *_poss_cols]].rename(
            columns={"frame": "frame_id", "period": "period_id"}
        )
        df = df.merge(poss_lu, on=["frame_id", "period_id"], how="left")
    else:
        for _c in _poss_cols:
            df[_c] = pd.NA
    # in_camera_view: per object row, computed in the SCALED (pre-orientation) frame. Orientation-invariant
    # (DL ruling): the scale is a positive-diagonal affine and the LTR flip is a 180 deg rotation, both preserve
    # containment and corner order. A degenerate / null footprint -> <NA>, counted on the report.
    _footprint = _scale_footprint(src, pitch_length=pitch_length, pitch_width=pitch_width)
    # Count only PRESENT-but-degenerate footprints (non-convex / zero-area). An ABSENT footprint (all-NaN
    # corners, e.g. a bronze that carries none) is not "degenerate" -- it simply has no footprint to test.
    _fp_present = _footprint[[*FOOTPRINT_X_COLUMNS, *FOOTPRINT_Y_COLUMNS]].notna().all(axis=1).to_numpy()
    n_degenerate_footprint = int((footprint_degenerate_mask(_footprint) & _fp_present).sum())
    _fp_rows = _footprint.set_index(["period_id", "frame_id"]).reindex(
        pd.MultiIndex.from_arrays([df["period_id"], df["frame_id"]], names=["period_id", "frame_id"])
    )
    _corners = np.stack(
        [
            np.stack([_fp_rows[c].to_numpy("float64") for c in FOOTPRINT_X_COLUMNS], axis=1),
            np.stack([_fp_rows[c].to_numpy("float64") for c in FOOTPRINT_Y_COLUMNS], axis=1),
        ],
        axis=2,
    )
    df["in_camera_view"] = in_camera_view_mask(df["x"].to_numpy("float64"), df["y"].to_numpy("float64"), _corners)

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

    df["z_source"] = Z_PROVENANCE["skillcorner"]  # ADR-125: estimated (broadcast tracking_extrapolated)

    final = pd.DataFrame({c: df[c] for c in SKILLCORNER_TRACKING_FRAMES_COLUMNS})
    for c, dt in SKILLCORNER_TRACKING_FRAMES_COLUMNS.items():
        if dt == "bool":
            final[c] = final[c].astype("bool")
        elif dt == "boolean":  # nullable boolean (in_camera_view): <NA>-preserving
            final[c] = final[c].astype("boolean")
        elif dt in {"int64", "float64", "float32"}:
            final[c] = pd.to_numeric(final[c], errors="coerce").astype(dt)  # type: ignore[arg-type]
        elif dt == "category":
            final[c] = final[c].astype("category")
        else:
            final[c] = final[c].astype(object)

    ltr_flip_by_period: dict[int, bool | None] = {}
    if output_convention == "ltr":
        if home_team_start_left is None:
            # Decide the per-period geometric flip ONCE, record it, thread it to the orienter (DL ruling: not
            # decided twice). The camera-footprint companion applies the SAME recorded flip.
            _is_ball = final["is_ball"].astype(bool).to_numpy()
            ltr_flip_by_period = _decide_ltr_flips(
                x_arr=final["x"].to_numpy("float64"),
                period_arr=final["period_id"].to_numpy(),
                home_arr=ids_match(final["team_id"], str(home_team_id)).fillna(False).to_numpy(dtype=bool),
                player_arr=~_is_ball,
                gk_arr=final["is_goalkeeper"].astype(bool).to_numpy(),
            )
            final = _orient_frames_to_ltr_by_geometry(
                final,
                home_team_id=str(home_team_id),
                source="skillcorner",
                game_id=game_id,
                on_missing_home="raise",
                copy=True,
                flips=ltr_flip_by_period,
                stacklevel=2,
            )
        else:
            # Flag path: flip = not home_attacks_right (spec section 17.7 amendment).
            _har = home_attacks_right_per_period(
                home_team_start_left=home_team_start_left,
                home_team_start_left_extratime=home_team_start_left_extratime,
            )
            _present = sorted({int(p) for p in pd.Series(final["period_id"]).dropna().unique()})
            ltr_flip_by_period = {p: (not _har[p]) for p in _present if p in _har}
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
        ltr_flip_by_period=ltr_flip_by_period,
        n_degenerate_footprint_frames=n_degenerate_footprint,
        game_id=game_id,
        provider_period_frame_ranges=provider_ranges,
    )
    # The coords are scaled to the 105 x 68 frame, so the ORIGINAL pitch dimensions (which fix where every
    # real-metre Law location lands after scaling: the spot, the goal area, the corner arc; spec sections 6.4 / 12,
    # amended 2026-10-22) are stamped here, after every transform (attrs do not survive most pandas ops, so set
    # them last, before return).
    final.attrs["pitch_length"] = float(pitch_length)
    final.attrs["pitch_width"] = float(pitch_width)
    # The flip APPLIED to these frames per period (R.2 2026-10-23): a GI attacking_side is in the delivered frame,
    # so the GI direction check maps it through this record. "absolute_frame" applies none (False); a "ltr" period
    # without a decision stays None (unresolved). A JSON-safe triple list: attrs cross parquet only as JSON. A
    # consumer should still pass frame_flips_from_report(report) explicitly -- attrs drop through most pandas
    # operations and every write to disk.
    applied = (
        ltr_flip_by_period
        if output_convention == "ltr"
        else {int(p): False for p in pd.Series(final["period_id"]).dropna().unique()}
    )
    final.attrs[FRAME_FLIPS_ATTR] = [[str(game_id), int(p), v] for p, v in sorted(applied.items())]
    if provider_ranges is not None:
        # The PROVIDER period range (spec section 17.7, amended 2026-10-25): a JSON-safe fallback for the GI pairing
        # Range test; pass period_frame_ranges_from_report(report) explicitly (attrs drop through most operations).
        final.attrs[PERIOD_FRAME_RANGES_ATTR] = [
            [str(game_id), int(p), int(lo), int(hi)] for p, (lo, hi) in sorted(provider_ranges.items())
        ]
    return final, report


def provider_period_frame_ranges(records: Iterable[Mapping[str, Any]] | pd.DataFrame) -> dict[int, tuple[int, int]]:
    """The PROVIDER period range of one SkillCorner match: ``{period_id: (first_frame, last_frame)}`` over every raw
    tracking record the provider labels with that period, empty frames (no objects) included, records with period
    ``None`` (before kick-off, the inter-period jump) excluded (spec sections 6.1 / 17.7, amended 2026-10-25). Pure.

    ``records`` is the raw feed as a list of frame records (``frame``, ``period``; the JSONL / parquet record shape)
    or a frame with those columns. The bronze cannot give this: it carries only frames with objects.

    Examples
    --------
    Compute the range from the raw records, then pass it with the bronze built from the same records::

        ranges = provider_period_frame_ranges(raw_records)        # {1: (10, 28530), 2: (28540, 58080)}
        frames, report = convert_to_frames(bronze, home_team_id="31", period_frame_ranges=ranges)
    """
    if isinstance(records, pd.DataFrame):
        pairs = zip(records["frame"].tolist(), records["period"].tolist(), strict=True)
    else:
        pairs = ((r.get("frame"), r.get("period")) for r in records)
    out: dict[int, tuple[int, int]] = {}
    for frame, period in pairs:
        if frame is None or period is None or pd.isna(frame) or pd.isna(period):
            continue
        p, f = int(period), int(frame)
        lo, hi = out.get(p, (f, f))
        out[p] = (min(lo, f), max(hi, f))
    return dict(sorted(out.items()))


def _check_period_frame_ranges(
    ranges: Mapping[int, tuple[int, int]] | None, src: pd.DataFrame
) -> dict[int, tuple[int, int]] | None:
    """Validate a supplied provider period range against the bronze: keyed by period (one match), and every
    retained frame inside its period's range (the retained frames are a subset of the raw ones). ``None`` gives
    ``None``."""
    if ranges is None:
        return None
    bad = [k for k in ranges if isinstance(k, tuple)]
    if bad:
        raise TypeError(
            f"convert_to_frames: period_frame_ranges is keyed by period_id for ONE match; got key {bad[0]!r}. "
            "Build it with provider_period_frame_ranges(raw_records)."
        )
    clean = {int(p): (int(lo), int(hi)) for p, (lo, hi) in ranges.items()}
    for p, g in src.groupby("period"):
        period = _as_int(p)
        lohi = clean.get(period)
        f = g["frame"].to_numpy("int64")
        if lohi is None or int(f.min()) < lohi[0] or int(f.max()) > lohi[1]:
            raise ValueError(
                f"convert_to_frames: retained period {period} frames {int(f.min())}..{int(f.max())} lie outside the "
                f"supplied provider range {lohi} -- the range belongs to another match or another feed."
            )
    return dict(sorted(clean.items()))


def _as_int(v) -> int:
    """pandas Scalar / Hashable -> int; the untyped arg sidesteps the stub friction (the ``_restart_frame._i``
    idiom)."""
    return int(v)


def period_frame_ranges_from_report(report: TrackingConversionReport) -> dict[tuple[str, int], tuple[int, int]]:
    """The PROVIDER period range per ``(game_id, period_id)`` -- the ``period_ranges`` input of
    ``assess_gi_pairing`` / ``resolve_kicks`` -- from the SkillCorner tracking report of ONE match (the CL-37
    ``frame_flips_from_report`` pattern; a period-only mapping passed downstream raises ``TypeError``). RAISES
    ``ValueError`` when the report is not a SkillCorner report, carries no ``game_id``, or recorded no provider
    range (``convert_to_frames`` was called without ``period_frame_ranges``).

    Examples
    --------
    Convert with the raw range, then pass THAT report's range to the pairing::

        frames, report = convert_to_frames(bronze, home_team_id="31", period_frame_ranges=ranges)
        period_ranges = period_frame_ranges_from_report(report)  # {("m1", 1): (10, 28530), ("m1", 2): (28540, 58080)}
    """
    if report.provider != "skillcorner":
        raise ValueError(
            f"period_frame_ranges_from_report: report.provider={report.provider!r}, expected 'skillcorner'."
        )
    if report.game_id is None:
        raise ValueError("period_frame_ranges_from_report: the report carries no game_id (the match cannot be keyed).")
    if report.provider_period_frame_ranges is None:
        raise ValueError(
            "period_frame_ranges_from_report: the report recorded no provider period range -- pass "
            "period_frame_ranges=provider_period_frame_ranges(raw_records) to convert_to_frames."
        )
    gid = str(report.game_id)
    return {(gid, int(p)): (int(lo), int(hi)) for p, (lo, hi) in sorted(report.provider_period_frame_ranges.items())}


def frame_flips_from_report(
    report: TrackingConversionReport,
    *,
    output_convention: Literal["absolute_frame", "ltr"] = "ltr",
) -> dict[tuple[str, int], bool | None]:
    """The frames' applied orientation flip per ``(game_id, period_id)`` -- the ``flips`` input of
    ``gi_direction_check`` / ``resolve_kicks`` -- from the SkillCorner tracking report of ONE match.

    ``report.ltr_flip_by_period`` is keyed by period only; this adds the report's ``game_id``. Pass the SAME
    ``output_convention`` the frames were converted with: ``"ltr"`` returns the recorded decision (a ``None``
    period stays ``None`` = unresolved); ``"absolute_frame"`` applied no flip, so every converted period is
    ``False``. RAISES ``ValueError`` when the report is not a SkillCorner report or carries no ``game_id`` (an
    unprovable match). Frames written to disk lose their ``attrs`` stamp, so store this record with them.

    Examples
    --------
    Convert the frames, then pass THAT report's flips to the GI direction check::

        frames, report = convert_to_frames(bronze, home_team_id="31")
        flips = frame_flips_from_report(report)  # {("m1", 1): False, ("m1", 2): True}
        check = gi_direction_check(actions, resolve_defended_goals(frames), flips=flips)
    """
    if report.provider != "skillcorner":
        raise ValueError(f"frame_flips_from_report: report.provider={report.provider!r}, expected 'skillcorner'.")
    if report.game_id is None:
        raise ValueError("frame_flips_from_report: the report carries no game_id (the match cannot be keyed).")
    gid = str(report.game_id)
    if output_convention == "absolute_frame":
        return {(gid, int(p)): False for p in sorted(report.frame_coverage_per_period)}
    return {(gid, int(p)): v for p, v in sorted(report.ltr_flip_by_period.items())}


def convert_camera_footprint(
    bronze: pd.DataFrame,
    *,
    report: TrackingConversionReport,
    assume_standard_pitch: bool = False,
) -> pd.DataFrame:
    """The SkillCorner camera-footprint companion table (spec section 17.7 / CL-15, DL ruling): one row per
    (game_id, period_id, frame_id) with the four footprint corners in the SPADL frame.

    It scales the footprint on the SAME pitch dims as ``convert_to_frames`` and APPLIES
    ``report.ltr_flip_by_period`` -- it NEVER re-decides the flip, so the footprint cannot disagree with the
    frames. It RAISES when ``report`` is not a SkillCorner report, or belongs to a different match (the frames
    and footprint must share one orientation decision).

    Examples
    --------
    Convert the frames first, then the footprint with THAT report::

        frames, report = convert_to_frames(bronze, home_team_id="31")
        footprint = convert_camera_footprint(bronze, report=report)
        # one row per (game_id, period_id, frame_id); cam_tl_x ... cam_bl_y in the frames' orientation
    """
    if report.provider != "skillcorner":
        raise ValueError(f"convert_camera_footprint: report.provider={report.provider!r}, expected 'skillcorner'.")
    game_id = str(bronze["match_id"].iloc[0])
    if report.game_id is None or str(report.game_id) != game_id:  # no game id = unprovable match = refuse
        raise ValueError(
            f"convert_camera_footprint: report.game_id={report.game_id!r} does not match the bronze "
            f"game_id={game_id!r} -- the report must belong to this match (its orientation decision applies)."
        )
    pitch_length, pitch_width = _resolve_pitch_dims(bronze, assume_standard_pitch)
    footprint = _scale_footprint(bronze, pitch_length=pitch_length, pitch_width=pitch_width)
    flips = {(game_id, int(p)): v for p, v in report.ltr_flip_by_period.items()}
    footprint = orient_camera_footprint(footprint, flips)
    out = pd.DataFrame({c: footprint[c] for c in SKILLCORNER_CAMERA_FOOTPRINT_COLUMNS})
    for c, dt in SKILLCORNER_CAMERA_FOOTPRINT_COLUMNS.items():
        if dt in {"int64", "float64", "float32"}:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(dt)  # type: ignore[arg-type]
        else:
            out[c] = out[c].astype(object)
    return out
