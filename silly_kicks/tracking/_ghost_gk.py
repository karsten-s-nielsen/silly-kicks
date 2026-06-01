"""Ghost-GK positioning model (TF-18, GKDV Layer 2).

Predicts where a league-average goalkeeper would position themselves
given the current game state, using RFCDE density estimation over
HistGradientBoostingRegressor leaf assignments.

Input frames MUST be in LTR-normalized convention (home team attacks
right in all periods — the standard silly-kicks tracking output after
play_left_to_right normalization).

See docs/superpowers/specs/2026-05-25-tf18-ghost-gk-design.md.
See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import gaussian_kde

from silly_kicks.spadl import config as spadlconfig

# ---------------------------------------------------------------------------
# Grid constants (fixed for API stability — see spec Density Grid)
# ---------------------------------------------------------------------------

GRID_X_MIN = 0.0
GRID_X_MAX = 30.0
GRID_Y_MIN = 18.0
GRID_Y_MAX = 50.0
GRID_NX = 60
GRID_NY = 64
GRID_RESOLUTION = 0.5  # meters per cell

# Cell centers
_GRID_X = np.linspace(
    GRID_X_MIN + GRID_RESOLUTION / 2,
    GRID_X_MAX - GRID_RESOLUTION / 2,
    GRID_NX,
)
_GRID_Y = np.linspace(
    GRID_Y_MIN + GRID_RESOLUTION / 2,
    GRID_Y_MAX - GRID_RESOLUTION / 2,
    GRID_NY,
)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GhostGkDensity:
    """Per-frame ghost-GK density prediction.

    Attributes
    ----------
    mode_x : float
        Joint 2D mode x (argmax of probabilities grid), goal-relative.
    mode_y : float
        Joint 2D mode y (argmax of probabilities grid), goal-relative.
    mean_x : float
        Density-weighted mean x.
    mean_y : float
        Density-weighted mean y.
    spread : float
        Effective area (entropy-based measure of density dispersion).
    grid_x : np.ndarray
        Shape (60,) — x-axis cell centers covering [0, 30].
    grid_y : np.ndarray
        Shape (64,) — y-axis cell centers covering [18, 50].
    probabilities : np.ndarray
        Shape (60, 64) — probability mass per cell, sums to ~1.0.
        Read-only (writeable=False).

    Examples
    --------
    >>> density = model.predict_density(features)[0]
    >>> density.mode_x, density.mode_y
    (5.25, 34.25)
    >>> density.probabilities.sum()
    1.0
    """

    mode_x: float
    mode_y: float
    mean_x: float
    mean_y: float
    spread: float
    grid_x: np.ndarray
    grid_y: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        # Prevent mutation of numpy arrays after construction
        object.__setattr__(self, "grid_x", self.grid_x.copy())
        object.__setattr__(self, "grid_y", self.grid_y.copy())
        object.__setattr__(self, "probabilities", self.probabilities.copy())
        self.grid_x.flags.writeable = False
        self.grid_y.flags.writeable = False
        self.probabilities.flags.writeable = False


# ---------------------------------------------------------------------------
# Errors + constants
# ---------------------------------------------------------------------------


class IntegrityError(Exception):
    """Raised when artifact SHA-256 verification fails."""


_ENV_VAR = "SILLY_KICKS_GHOST_GK_PATH"
_WEIGHTS_ROOT = Path(__file__).parent / "_ghost_gk_weights"

#: Valid model variant names for the ``model`` parameter.
GhostGkVariant = Literal["default", "full"]


_HF_REPO_ID = "silly-kicks/ghost-gk-v1"


def _resolve_model(model: GhostGkModel | GhostGkVariant | None) -> GhostGkModel:
    """Resolve model parameter with cascade: caller > env > bundled/Hub variant.

    Resolution order:
    1. Caller-supplied ``GhostGkModel`` instance (pass-through)
    2. ``SILLY_KICKS_GHOST_GK_PATH`` env var (custom-trained model path)
    3. Bundled variant by name — ``"default"`` is bundled in the wheel;
       ``"full"`` is downloaded from HuggingFace Hub on first use
       (requires ``pip install silly-kicks[ghost-gk]``).

    Parameters
    ----------
    model : GhostGkModel | "default" | "full" | None
        - ``None`` or ``"default"``: lightweight model (~9 MB, 36 k samples).
          Bundled in the wheel — works offline, no download needed.
        - ``"full"``: high-resolution model (~91 MB, 537 k samples).
          Downloaded from HuggingFace Hub on first use; cached locally.
          Smoother density surfaces at the cost of slower ``predict_density``.
        - ``GhostGkModel``: pre-loaded instance, returned as-is.

    Examples
    --------
    >>> resolved = _resolve_model(None)  # default bundled weights
    >>> resolved = _resolve_model("full")  # download from HuggingFace Hub
    >>> resolved = _resolve_model(my_model)  # pass-through
    """
    if isinstance(model, GhostGkModel):
        return model

    # Check env var override
    env_path = os.environ.get(_ENV_VAR)
    if env_path is not None:
        return GhostGkModel.load(Path(env_path))

    # Resolve variant name
    variant: GhostGkVariant = model if model is not None else "default"

    # Try bundled weights first (default is always bundled; full may be
    # present in a dev checkout but is not shipped in the wheel)
    weights_dir = _WEIGHTS_ROOT / variant
    if (weights_dir / "SHA256SUMS").exists():
        return GhostGkModel.load(weights_dir)

    # For "full" variant, download from HuggingFace Hub
    if variant == "full":
        return GhostGkModel.from_hub(_HF_REPO_ID)

    msg = f"Bundled Ghost-GK weights not found at {weights_dir}"
    raise FileNotFoundError(msg)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

_FIELD_LENGTH = spadlconfig.field_length  # 105.0
_FIELD_WIDTH = spadlconfig.field_width  # 68.0
_GOAL_Y = _FIELD_WIDTH / 2.0  # 34.0
_PENALTY_AREA_X = 16.5
_PENALTY_AREA_Y_MIN = (_FIELD_WIDTH - 40.3) / 2.0
_PENALTY_AREA_Y_MAX = (_FIELD_WIDTH + 40.3) / 2.0
_VELOCITY_WINDOW_S = 0.5
_SET_PIECE_DECAY_SECONDS = 10.0

GHOST_GK_FEATURE_NAMES: list[str] = [
    "ball_x",
    "ball_y",
    "ball_vx",
    "ball_vy",
    "ball_distance_to_goal",
    "defensive_line_x",
    "defensive_line_depth",
    "defensive_line_width",
    "attackers_in_box",
    "nearest_attacker_to_goal_x",
    "attacker_centroid_x",
    "attacker_centroid_y",
    "defenders_behind_ball",
    "deepest_defender_x",
    "phase",
    "team_in_possession",
    "score_diff",
    "time_seconds",
    "period_id",
    "ball_to_goal_angle",
    "ball_to_nearest_attacker_dist",
    "defending_team_compactness",
    "ball_in_own_half",
    "ball_speed",
    "defensive_line_speed",
    "defending_centroid_vx",
]


# ---------------------------------------------------------------------------
# Match context resolution
# ---------------------------------------------------------------------------


def _build_score_lookup(
    actions: pd.DataFrame,
    home_team_id: int | str,
) -> Callable[[Any, float], float]:
    """Build (game_id, time_seconds) -> score_diff callback (home perspective).

    Returns home_score - away_score at the queried time. The **caller**
    must negate for away-team GKs.

    Own goals (result_name == "owngoal") are attributed to the opponent
    of the acting team.

    Note: Uses str() comparison internally for team ID matching.
    Assumes actions and frames share team ID type from the same provider
    (always true in practice --- both come from the same kloppy/converter
    pipeline). If actions have int team_id=1 and caller passes
    home_team_id="001", str(1) != "001" would mismatch.

    Examples
    --------
    >>> fn = _build_score_lookup(actions, home_team_id=1)
    >>> fn("100", 30.0)
    1.0
    """

    # Resolve type/result columns (supports both ID and name DataFrames)
    if "type_name" in actions.columns:
        shots = actions[actions["type_name"] == "shot"].copy()
    else:
        shot_id = spadlconfig.actiontype_id["shot"]
        shots = actions[actions["type_id"] == shot_id].copy()

    if "result_name" in actions.columns:
        goals = shots[shots["result_name"].isin(["success", "owngoal"])].copy()
    else:
        success_id = spadlconfig.result_id["success"]
        owngoal_id = spadlconfig.result_id["owngoal"]
        goals = shots[shots["result_id"].isin([success_id, owngoal_id])].copy()

    if len(goals) == 0:

        def _zero(_game_id: Any, _time_s: float) -> float:
            return 0.0

        return _zero

    # Flip own-goal team attribution
    if "result_name" in goals.columns:
        is_own = goals["result_name"] == "owngoal"
    else:
        is_own = goals["result_id"] == spadlconfig.result_id["owngoal"]

    # For own goals, the scoring team is the OPPONENT of the actor
    goals = goals.copy()
    goals["_scoring_team"] = goals["team_id"].copy()
    if is_own.any():
        all_teams = actions["team_id"].unique()
        if len(all_teams) == 2:
            team_a, team_b = all_teams[0], all_teams[1]
            flip_map = {team_a: team_b, team_b: team_a}
            goals.loc[is_own, "_scoring_team"] = goals.loc[is_own, "team_id"].map(flip_map)

    goals = goals.sort_values(["game_id", "time_seconds"]).reset_index(drop=True)

    # Build per-game cumulative score arrays
    home_team_id_norm = str(home_team_id)
    _lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for gid, grp in goals.groupby("game_id"):
        times = np.asarray(grp["time_seconds"].values, dtype=np.float64)
        is_home = np.array([str(t) == home_team_id_norm for t in grp["_scoring_team"]])
        home_cum = np.cumsum(is_home.astype(np.float64))
        away_cum = np.cumsum((~is_home).astype(np.float64))
        diffs = home_cum - away_cum
        _lookup[str(gid)] = (times, diffs)

    def _score(game_id: Any, time_s: float) -> float:
        key = str(game_id)
        if key not in _lookup:
            return 0.0
        times, diffs = _lookup[key]
        idx = int(np.searchsorted(times, time_s, side="right")) - 1
        if idx < 0:
            return 0.0
        return float(diffs[idx])

    return _score


def _build_phase_lookup(
    actions: pd.DataFrame,
) -> Callable[[Any, float], int]:
    """Build (game_id, time_seconds) -> phase callback.

    Returns 0 (open_play), 1 (set_piece), or 2 (goal_kick).
    A set-piece phase decays to open play after _SET_PIECE_DECAY_SECONDS.
    throw_in is excluded --- does not alter GK positioning expectations.

    Examples
    --------
    >>> fn = _build_phase_lookup(actions)
    >>> fn("100", 33.0)
    1
    """

    # Set-piece types (excluding throw_in per spec)
    _SP_TYPES = {"freekick_crossed", "freekick_short", "corner_crossed", "corner_short"}
    _GK_TYPE = "goalkick"

    # Resolve type column
    if "type_name" in actions.columns:
        sp_mask = actions["type_name"].isin(_SP_TYPES | {_GK_TYPE})
        sp = actions[sp_mask].copy()
        if len(sp) > 0:
            sp["_phase_code"] = sp["type_name"].apply(lambda t: 2 if t == _GK_TYPE else 1)
        else:
            sp["_phase_code"] = pd.Series(dtype=int)
    else:
        sp_ids = {spadlconfig.actiontype_id[t] for t in _SP_TYPES if t in spadlconfig.actiontype_id}
        gk_id = spadlconfig.actiontype_id.get(_GK_TYPE)
        if gk_id is not None:
            sp_ids.add(gk_id)
        sp = actions[actions["type_id"].isin(sp_ids)].copy()
        if len(sp) > 0:
            sp["_phase_code"] = sp["type_id"].apply(lambda tid: 2 if tid == gk_id else 1)
        else:
            sp["_phase_code"] = pd.Series(dtype=int)

    if len(sp) == 0:

        def _open(_game_id: Any, _time_s: float) -> int:
            return 0

        return _open

    sp = sp.sort_values(["game_id", "time_seconds"]).reset_index(drop=True)

    _lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for gid, grp in sp.groupby("game_id"):
        times = np.asarray(grp["time_seconds"].values, dtype=np.float64)
        codes = np.asarray(grp["_phase_code"].values, dtype=np.int64)
        _lookup[str(gid)] = (times, codes)

    def _phase(game_id: Any, time_s: float) -> int:
        key = str(game_id)
        if key not in _lookup:
            return 0
        times, codes = _lookup[key]
        idx = int(np.searchsorted(times, time_s, side="right")) - 1
        if idx < 0:
            return 0
        elapsed = time_s - times[idx]
        if elapsed > _SET_PIECE_DECAY_SECONDS:
            return 0
        return int(codes[idx])

    return _phase


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def extract_ghost_gk_features(
    frame_data: pd.DataFrame,
    *,
    gk_team_id: int | str,
    goal_x: float = 0.0,
    score_diff: float = 0.0,
    phase: int = 0,
    ball_carrier_team_id: int | str | None = None,
    prev_defensive_line_x: float | None = None,
    prev_defending_centroid_x: float | None = None,
    dt: float = _VELOCITY_WINDOW_S,
) -> pd.DataFrame:
    """Extract 26 ghost-GK features from frame data in goal-relative coords.

    Accepts a single frame (one game_id/period_id/frame_id group) and
    returns a single-row DataFrame. Called in a vectorized batch by
    compute_ghost_gk.

    Input frames MUST be in LTR-normalized convention (home attacks right
    in all periods). The goal_x parameter specifies which end the GK
    defends: 0.0 for the home GK, 105.0 for the away GK.

    Parameters
    ----------
    frame_data : pd.DataFrame
        All rows for one frame — players + ball.
    gk_team_id : int | str
        Team ID of the GK whose ghost position we predict.
    goal_x : float
        x-coordinate of the defending goal (0.0 or 105.0).
    score_diff : float
        GK's team score minus opponent.
    phase : int
        0 = open_play, 1 = set_piece, 2 = goal_kick.
    ball_carrier_team_id : int | str | None
        Team currently in possession.
    prev_defensive_line_x : float | None
        Previous frame's defensive line x (for velocity).
    prev_defending_centroid_x : float | None
        Previous frame's defending centroid x.
    dt : float
        Time delta for velocity computation.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with 26 columns (GHOST_GK_FEATURE_NAMES).

    Examples
    --------
    >>> features = extract_ghost_gk_features(frame, gk_team_id=1, goal_x=0.0)
    >>> features.shape
    (1, 26)
    """
    # --- Coordinate transform ---
    flip = goal_x > 50.0

    def to_gr_x(x: float) -> float:
        return (_FIELD_LENGTH - x) if flip else x

    def to_gr_vx(vx: float) -> float:
        return -vx if flip else vx

    # --- Ball ---
    ball = frame_data[frame_data["is_ball"].astype(bool)]
    if len(ball) > 0:
        bx_raw = float(ball["x"].iloc[0])
        by_raw = float(ball["y"].iloc[0])
        bvx_raw = float(ball["vx"].iloc[0]) if "vx" in ball.columns else np.nan
        bvy_raw = float(ball["vy"].iloc[0]) if "vy" in ball.columns else np.nan
    else:
        bx_raw = by_raw = bvx_raw = bvy_raw = np.nan

    ball_x = to_gr_x(bx_raw)
    ball_y = by_raw
    ball_vx = to_gr_vx(bvx_raw)
    ball_vy = bvy_raw
    ball_speed = float(np.sqrt(bvx_raw**2 + bvy_raw**2)) if not np.isnan(bvx_raw) else np.nan
    ball_dist = float(np.sqrt(ball_x**2 + (ball_y - _GOAL_Y) ** 2)) if not np.isnan(ball_x) else np.nan

    # --- Player splits ---
    players = frame_data[~frame_data["is_ball"].astype(bool)]
    defending = players[(players["team_id"] == gk_team_id) & (~players["is_goalkeeper"].astype(bool))]
    attacking = players[(players["team_id"] != gk_team_id) & (~players["is_goalkeeper"].astype(bool))]
    gk_rows = players[(players["team_id"] == gk_team_id) & (players["is_goalkeeper"].astype(bool))]

    # --- Defensive line ---
    if len(defending) > 0:
        def_xs = np.asarray(defending["x"].apply(to_gr_x).values)
        sorted_xs = np.sort(def_xs)
        n_back = min(4, len(sorted_xs))
        defensive_line_x = float(np.median(sorted_xs[:n_back]))
        deepest_defender_x = float(sorted_xs[0])
        defensive_line_width = float(defending["y"].max() - defending["y"].min())
    else:
        defensive_line_x = deepest_defender_x = defensive_line_width = np.nan

    if len(gk_rows) > 0 and not np.isnan(defensive_line_x):
        gk_x_gr = to_gr_x(float(gk_rows["x"].iloc[0]))
        defensive_line_depth = abs(defensive_line_x - gk_x_gr)
    else:
        defensive_line_depth = np.nan

    # --- Attacking players ---
    if len(attacking) > 0:
        atk_xs = np.asarray(attacking["x"].apply(to_gr_x).values)
        atk_ys = np.asarray(attacking["y"].values)
        nearest_atk_x = float(np.min(atk_xs))
        atk_cx = float(np.mean(atk_xs))
        atk_cy = float(np.mean(atk_ys))
        in_box = (atk_xs < _PENALTY_AREA_X) & (atk_ys >= _PENALTY_AREA_Y_MIN) & (atk_ys <= _PENALTY_AREA_Y_MAX)
        attackers_in_box = int(np.sum(in_box))
    else:
        nearest_atk_x = atk_cx = atk_cy = np.nan
        attackers_in_box = 0

    # --- Defenders behind ball ---
    if len(defending) > 0 and not np.isnan(ball_x):
        def_xs_arr = np.asarray(defending["x"].apply(to_gr_x).values)
        defenders_behind_ball = int(np.sum(def_xs_arr < ball_x))
    else:
        defenders_behind_ball = 0

    # --- Spatial geometry ---
    ball_to_goal_angle = float(np.arctan2(ball_y - _GOAL_Y, ball_x)) if not np.isnan(ball_x) else np.nan

    if len(attacking) > 0 and not np.isnan(ball_x):
        atk_xs_rel = np.asarray(attacking["x"].apply(to_gr_x).values)
        dists = np.sqrt((atk_xs_rel - ball_x) ** 2 + (np.asarray(attacking["y"].values) - ball_y) ** 2)
        ball_to_nearest_atk = float(np.min(dists))
    else:
        ball_to_nearest_atk = np.nan

    if len(defending) >= 3:
        coords = np.column_stack([np.asarray(defending["x"].apply(to_gr_x).values), np.asarray(defending["y"].values)])
        try:
            hull = ConvexHull(coords)
            compactness = float(hull.volume)
        except QhullError:
            compactness = np.nan
    else:
        compactness = np.nan

    ball_in_own_half = 1.0 if (not np.isnan(ball_x) and ball_x < _FIELD_LENGTH / 2) else 0.0
    try:
        team_in_poss = 1.0 if ball_carrier_team_id is not None and ball_carrier_team_id == gk_team_id else 0.0
    except (ValueError, TypeError):
        # pd.NA comparison raises TypeError; NaN comparison is always False
        team_in_poss = 0.0
    period_clamped = min(int(frame_data["period_id"].iloc[0]), 2)
    time_s = float(frame_data["time_seconds"].iloc[0]) if "time_seconds" in frame_data.columns else 0.0

    # --- Velocity (actual dt from consecutive frames) ---
    if prev_defensive_line_x is not None and not np.isnan(defensive_line_x):
        def_line_speed = (defensive_line_x - prev_defensive_line_x) / dt
    else:
        def_line_speed = np.nan

    if prev_defending_centroid_x is not None and len(defending) > 0:
        def_cx = float(np.mean(np.asarray(defending["x"].apply(to_gr_x).values)))
        def_centroid_vx = (def_cx - prev_defending_centroid_x) / dt
    else:
        def_centroid_vx = np.nan

    row = [
        ball_x,
        ball_y,
        ball_vx,
        ball_vy,
        ball_dist,
        defensive_line_x,
        defensive_line_depth,
        defensive_line_width,
        attackers_in_box,
        nearest_atk_x,
        atk_cx,
        atk_cy,
        defenders_behind_ball,
        deepest_defender_x,
        phase,
        team_in_poss,
        score_diff,
        time_s,
        period_clamped,
        ball_to_goal_angle,
        ball_to_nearest_atk,
        compactness,
        ball_in_own_half,
        ball_speed,
        def_line_speed,
        def_centroid_vx,
    ]
    return pd.DataFrame([row], columns=GHOST_GK_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Shared batch helper
# ---------------------------------------------------------------------------


def _extract_all_ghost_gk_features(
    frames: pd.DataFrame,
    *,
    home_team_id: str | int,
    carrier: pd.DataFrame | None = None,
    score_at_time: Callable[[Any, float], float] | None = None,
    phase_at_time: Callable[[Any, float], int] | None = None,
    subsample_fps: float | None = None,
    link_frame_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shared batch helper: iterate frames, extract features for every GK.

    Both compute_ghost_gk (inference) and prepare_ghost_gk_training_data
    (training) call this function. Single source of truth for the
    frame-iteration + velocity-tracking + feature-extraction loop.

    Parameters
    ----------
    frames : pd.DataFrame
        TRACKING_FRAMES_COLUMNS, LTR-normalized, vx/vy present.
    home_team_id : str | int
        Home team (used as fallback; defending goal is inferred per-period
        from mean GK x position to handle LTR data with period flips).
    carrier : pd.DataFrame | None
        Per-frame ball_carrier_team_id (from derive_team_in_possession).
    score_at_time : callable | None
        (game_id, time_seconds) -> score_diff (home perspective).
    phase_at_time : callable | None
        (game_id, time_seconds) -> int (0=open, 1=set_piece, 2=goal_kick).
    subsample_fps : float | None
        Thin frames to target fps before extraction.
    link_frame_ids : set[int] | None, default None
        When provided, build a feature row ONLY for frames in this set, but still
        walk every frame to maintain the cross-period one-step velocity state, so
        each linked frame sees its true predecessor (byte-identical velocity to the
        unrestricted pass). The per-period defending-goal mean-x is computed over
        the full frames either way. When None, every frame is extracted. PR-S66 §5.

    Returns
    -------
    features : pd.DataFrame
        (n_samples, len(GHOST_GK_FEATURE_NAMES)).
    meta : pd.DataFrame
        (n_samples, 6): game_id, period_id, frame_id, gk_team_id,
        gk_x_gr, gk_y_gr.

    Examples
    --------
    >>> features, meta = _extract_all_ghost_gk_features(frames, home_team_id=1)
    >>> features.shape[1]
    26
    """
    # --- Team ID normalization (§7) ---
    frame_team_dtype = frames["team_id"].dtype
    if frame_team_dtype is object:
        home_team_id = str(home_team_id)
    else:
        try:
            home_team_id = type(frames["team_id"].dropna().iloc[0])(home_team_id)
        except (ValueError, TypeError) as exc:
            raise TypeError(
                f"home_team_id={home_team_id!r} cannot be coerced to frames['team_id'] dtype {frame_team_dtype}"
            ) from exc

    # --- Pre-index carrier for O(1) lookup ---
    carrier_idx: pd.Series | None = None
    if carrier is not None and "ball_carrier_team_id" in carrier.columns:
        carrier_idx = carrier.set_index(["game_id", "period_id", "frame_id"])["ball_carrier_team_id"]

    # --- Subsample ---
    work = frames
    if subsample_fps is not None and "frame_rate" in frames.columns:
        fr = frames["frame_rate"].iloc[0]
        if fr > 0 and subsample_fps > 0:
            step = max(1, round(fr / subsample_fps))
            # Keep every step-th unique frame_id per (game_id, period_id)
            unique_frames = (
                frames[["game_id", "period_id", "frame_id"]]
                .drop_duplicates()
                .sort_values(["game_id", "period_id", "frame_id"])
            )
            keep_mask = unique_frames.groupby(["game_id", "period_id"]).cumcount() % step == 0
            keep_keys = unique_frames[keep_mask.values]
            work = frames.merge(keep_keys, on=["game_id", "period_id", "frame_id"])

    # --- Precompute defending goal per (game_id, period_id, team_id) ---
    # On LTR-normalized data with period flips (e.g. SkillCorner), teams swap
    # ends at halftime.  Using team identity alone to assign goal_x is wrong
    # for the flipped period.  Instead, use the GK's mean x per period to
    # determine which goal the GK defends: mean_x < 52.5 → defending x=0,
    # otherwise defending x=105.
    _gk_mask = work["is_goalkeeper"].astype(bool) & ~work["is_ball"].astype(bool)
    _gk_mean_x = work[_gk_mask].groupby(["game_id", "period_id", "team_id"])["x"].mean()
    _defending_goal: dict = {
        key: 0.0 if mean_x < _FIELD_LENGTH / 2 else _FIELD_LENGTH for key, mean_x in _gk_mean_x.items()
    }

    # --- Group and iterate ---
    group_keys = ["game_id", "period_id", "frame_id"]
    grouped = list(work.groupby(group_keys, sort=True))

    feature_rows: list[pd.DataFrame] = []
    meta_rows: list[dict] = []
    prev_state: dict[tuple, tuple[float, float]] = {}
    prev_timestamps: dict[tuple, float] = {}

    for (gid, pid, fid), frame_data in grouped:
        gk_rows = frame_data[frame_data["is_goalkeeper"].astype(bool) & ~frame_data["is_ball"].astype(bool)]
        time_s = float(frame_data["time_seconds"].iloc[0]) if "time_seconds" in frame_data.columns else 0.0

        # PR-S66 §5: when link_frame_ids is set, build a feature row only for
        # linked frames, but ALWAYS walk every frame below to update the velocity
        # state so a linked frame's predecessor is its true (cross-period) neighbour
        # rather than the previous *linked* frame.
        # fid is the groupby key (int/np.int); membership in the int set matches
        # by value regardless of numpy-vs-python int, so no explicit cast needed.
        fid_linked = link_frame_ids is None or fid in link_frame_ids

        for _, gk_row in gk_rows.iterrows():
            gk_team = gk_row["team_id"]
            goal_x = _defending_goal.get((gid, pid, gk_team), 0.0 if gk_team == home_team_id else _FIELD_LENGTH)
            flip = goal_x > 50.0

            # Cheap defensive-line-x + centroid in goal-relative coords, computed
            # for EVERY frame to drive the velocity state. These mirror exactly
            # extract_ghost_gk_features' defensive_line_x and the stored centroid
            # (median of the back-4 goal-relative x; mean goal-relative x) — see
            # TestExtractionRestriction golden which guards bit-identical velocity.
            defending = frame_data[
                (frame_data["team_id"] == gk_team)
                & ~frame_data["is_goalkeeper"].astype(bool)
                & ~frame_data["is_ball"].astype(bool)
            ]
            if len(defending) > 0:
                _dxs = np.asarray(defending["x"].values)
                _gr = (_FIELD_LENGTH - _dxs) if flip else _dxs
                _sorted_xs = np.sort(_gr)
                _n_back = min(4, len(_sorted_xs))
                dl_x = float(np.median(_sorted_xs[:_n_back]))
                def_cx = float(np.mean(_gr))
            else:
                dl_x = np.nan
                def_cx = np.nan

            # Velocity state (true predecessor regardless of restriction)
            state_key = (gid, gk_team)
            prev_dl_x, prev_dc_x = prev_state.get(state_key, (None, None))
            prev_ts = prev_timestamps.get(state_key)
            actual_dt = (time_s - prev_ts) if prev_ts is not None and time_s > prev_ts else _VELOCITY_WINDOW_S

            if fid_linked:
                # Score: callback returns home perspective, negate for away
                if score_at_time is not None:
                    sd = score_at_time(gid, time_s)
                    if gk_team != home_team_id:
                        sd = -sd
                else:
                    sd = 0.0

                # Phase
                ph = phase_at_time(gid, time_s) if phase_at_time is not None else 0

                # Carrier
                carrier_team = None
                if carrier_idx is not None:
                    key = (gid, pid, fid)
                    if key in carrier_idx.index:
                        carrier_team = carrier_idx[key]  # type: ignore[call-overload]

                feat = extract_ghost_gk_features(
                    frame_data,
                    gk_team_id=gk_team,
                    goal_x=goal_x,
                    score_diff=sd,
                    phase=ph,
                    ball_carrier_team_id=carrier_team,
                    prev_defensive_line_x=prev_dl_x,
                    prev_defending_centroid_x=prev_dc_x,
                    dt=actual_dt,
                )
                feature_rows.append(feat)

                # GK position in goal-relative coords for labels
                gk_x_raw = float(gk_row["x"])
                gk_y_raw = float(gk_row["y"])
                gk_x_gr = (_FIELD_LENGTH - gk_x_raw) if flip else gk_x_raw
                gk_y_gr = gk_y_raw

                meta_rows.append(
                    {
                        "game_id": gid,
                        "period_id": pid,
                        "frame_id": fid,
                        "gk_team_id": gk_team,
                        "gk_x_gr": gk_x_gr,
                        "gk_y_gr": gk_y_gr,
                    }
                )

            # Update velocity state (every frame, linked or not)
            prev_state[state_key] = (dl_x, def_cx)
            prev_timestamps[state_key] = time_s

    if not feature_rows:
        return (
            pd.DataFrame(columns=GHOST_GK_FEATURE_NAMES),
            pd.DataFrame(columns=["game_id", "period_id", "frame_id", "gk_team_id", "gk_x_gr", "gk_y_gr"]),
        )

    features = pd.concat(feature_rows, ignore_index=True)
    meta = pd.DataFrame(meta_rows)
    return features, meta


def prepare_ghost_gk_training_data(
    frames: pd.DataFrame,
    *,
    home_team_id: str | int,
    actions: pd.DataFrame | None = None,
    subsample_fps: float | None = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble training features + labels from one game's tracking frames.

    Parameters
    ----------
    frames : pd.DataFrame
        TRACKING_FRAMES_COLUMNS schema, LTR-normalized, with vx/vy
        columns (from smooth_frames + derive_velocities).
    home_team_id : str | int
        Home team ID (attacks right in LTR convention).
    actions : pd.DataFrame | None
        SPADL actions for the same game. Provides score_diff and phase
        context. If None, both default to 0 (valid but less informative).
    subsample_fps : float | None
        Target frame rate for training (default 1.0 Hz). None = no
        subsampling.

    Returns
    -------
    features : pd.DataFrame
        (n_samples, len(GHOST_GK_FEATURE_NAMES)) with GHOST_GK_FEATURE_NAMES
        columns.
    labels : pd.DataFrame
        (n_samples, 2) with columns "gk_x", "gk_y" in goal-relative
        coordinates matching the GhostGkModel training domain
        ([0, 30] x [18, 50]).

    Examples
    --------
    >>> features, labels = prepare_ghost_gk_training_data(
    ...     frames, home_team_id=1, actions=actions, subsample_fps=1.0
    ... )
    >>> model = GhostGkModel()
    >>> model.fit(features, labels)
    """
    import warnings

    from ._ball_carrier import infer_ball_carrier

    # Build context callbacks
    score_fn = _build_score_lookup(actions, home_team_id) if actions is not None else None
    phase_fn = _build_phase_lookup(actions) if actions is not None else None

    # Carrier (always computed --- only needs frames)
    carrier_raw = infer_ball_carrier(frames)
    carrier_cols = carrier_raw[["game_id", "period_id", "frame_id", "ball_carrier_team_id"]]

    features, meta = _extract_all_ghost_gk_features(
        frames,
        home_team_id=home_team_id,
        carrier=carrier_cols,
        score_at_time=score_fn,
        phase_at_time=phase_fn,
        subsample_fps=subsample_fps,
    )

    if len(meta) == 0:
        return (
            pd.DataFrame(columns=GHOST_GK_FEATURE_NAMES),
            pd.DataFrame(columns=["gk_x", "gk_y"]),
        )

    # Extract labels
    labels = meta[["gk_x_gr", "gk_y_gr"]].rename(columns={"gk_x_gr": "gk_x", "gk_y_gr": "gk_y"})

    # Drop NaN labels (GK not visible)
    valid = labels["gk_x"].notna() & labels["gk_y"].notna()
    features = features[valid.values].reset_index(drop=True)
    labels = labels[valid.values].reset_index(drop=True)

    # Validate feature width
    if features.shape[1] != len(GHOST_GK_FEATURE_NAMES):
        raise ValueError(f"Expected {len(GHOST_GK_FEATURE_NAMES)} features, got {features.shape[1]}")

    # Filter label domain (sweeper-keeper rushes, off-pitch artifacts)
    in_domain = (
        (labels["gk_x"] >= GRID_X_MIN)
        & (labels["gk_x"] <= GRID_X_MAX)
        & (labels["gk_y"] >= GRID_Y_MIN)
        & (labels["gk_y"] <= GRID_Y_MAX)
    )
    n_out = int((~in_domain).sum())
    if n_out > 0:
        total = len(labels)
        warnings.warn(
            f"Dropped {n_out} of {total} rows with GK outside goal-relative domain (sweeper rushes/artifacts)",
            stacklevel=2,
        )
        features = features[in_domain.values].reset_index(drop=True)
        labels = labels[in_domain.values].reset_index(drop=True)

    return features, labels


# ---------------------------------------------------------------------------
# Vectorized tree traversal
# ---------------------------------------------------------------------------


def _vectorized_leaf_indices(nodes_list: list[np.ndarray], X: np.ndarray) -> np.ndarray:
    """Vectorized tree traversal for all trees at once.

    Parameters
    ----------
    nodes_list : list[np.ndarray]
        Each element is a structured array of tree nodes with fields:
        left, right, feature_idx, num_threshold, missing_go_to_left.
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Shape (n_samples, n_trees) — leaf node index per sample per tree.
    """
    n_samples = X.shape[0]
    n_trees = len(nodes_list)
    leaves = np.zeros((n_samples, n_trees), dtype=np.intp)

    for t_idx, nodes in enumerate(nodes_list):
        current = np.zeros(n_samples, dtype=np.intp)

        for _ in range(100):  # depth bound
            node_data = nodes[current]
            is_leaf = node_data["left"] == 0
            if np.all(is_leaf):
                break

            feature_idx = node_data["feature_idx"]
            thresholds = node_data["num_threshold"]
            go_left_missing = node_data["missing_go_to_left"]

            # Gather feature values for each sample's current split feature
            feat_vals = X[np.arange(n_samples), feature_idx]

            # Decision: NaN handling + threshold comparison
            is_nan = np.isnan(feat_vals)
            go_left = np.where(
                is_nan,
                go_left_missing.astype(bool),
                feat_vals <= thresholds,
            )

            # Advance non-leaf samples
            next_node = np.where(go_left, node_data["left"], node_data["right"])
            current = np.where(is_leaf, current, next_node)

        leaves[:, t_idx] = current

    return leaves


def _leaf_match_weights(
    training_leaves: np.ndarray,
    query_leaves: np.ndarray,
    *,
    query_block: int = 64,
) -> np.ndarray:
    """Per-query leaf-match weight = fraction of trees with matching leaf.

    training_leaves : (n_train, n_trees); query_leaves : (n_query, n_trees).
    Returns (n_query, n_train) weights. Streams queries in blocks to bound the
    (qb, n_train, n_trees) broadcast for the 537k-train variant.
    """
    n_train, n_trees = training_leaves.shape
    n_query = query_leaves.shape[0]
    out = np.empty((n_query, n_train), dtype=np.float64)
    for start in range(0, n_query, query_block):
        sl = slice(start, min(start + query_block, n_query))
        eq = training_leaves[None, :, :] == query_leaves[sl][:, None, :]  # (qb, n_train, n_trees)
        out[sl] = eq.sum(axis=2).astype(np.float64) / n_trees
    return out


def _kde_density_scipy(
    gk_x_w: np.ndarray,
    gk_y_w: np.ndarray,
    w: np.ndarray,
    grid_points: np.ndarray,
) -> np.ndarray:
    """Reference per-sample weighted KDE on the fixed grid (scipy oracle).

    gk_x_w, gk_y_w, w : shape (k,) — the sample's nonzero-weight leaf subset.
    grid_points : shape (2, GRID_NX*GRID_NY).
    Returns the UNnormalized density grid (GRID_NX, GRID_NY); callers normalize.
    """
    kde = gaussian_kde(np.vstack([gk_x_w, gk_y_w]), weights=w, bw_method="scott")
    density_vals = kde(grid_points)
    return density_vals.reshape(GRID_NX, GRID_NY)


def _kde_density_vectorized(
    gk_x_w: np.ndarray,
    gk_y_w: np.ndarray,
    w: np.ndarray,
    grid_points: np.ndarray,
    *,
    train_block: int = 1024,
) -> np.ndarray:
    """Vectorized weighted Gaussian KDE on the fixed grid (scipy-faithful).

    Reuses scipy's Scott bandwidth + weighted covariance + Cholesky whitening
    (NOT a matrix inverse) so it matches scipy.stats.gaussian_kde within ~1e-9.
    Streams the training subset in blocks of ``train_block`` to bound memory for
    the 537k "full" variant. Returns the UNnormalized density grid (GRID_NX, GRID_NY).

    Default ``train_block=1024`` is the conservative serverless choice: with grid m=3840,
    the largest transient ``(2, kb, m)`` + tdiff + energy is ~150 MB/block, safe under the
    Databricks 1 GB ``applyInPandas`` cap. The serverless venue runs the 9 MB "default" model
    whose per-sample nonzero leaf subsets are small (k often < 1024 -> chunking rarely binds),
    so 1024 costs ~nothing there; the local benchmark can raise it for the in-memory "full"
    model.
    """
    from scipy.linalg import cho_factor, cho_solve

    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    data = np.vstack([np.asarray(gk_x_w, np.float64), np.asarray(gk_y_w, np.float64)])  # (2, k)
    d = 2
    neff = 1.0 / np.sum(w**2)
    factor = neff ** (-1.0 / (d + 4))
    data_cov = np.atleast_2d(np.cov(data, rowvar=True, bias=False, aweights=w))
    covariance = data_cov * factor**2
    # Cholesky path (matches scipy _kde.py): whitening via cho_solve; lower=True so the
    # factor's diagonal is in np.diag(chol[0]). cho_factor raises np.linalg.LinAlgError on a
    # singular covariance (collinear/identical points) -- same as scipy's cholesky -- and
    # predict_density's `except np.linalg.LinAlgError` degrades both to the uniform grid.
    chol = cho_factor(covariance, lower=True)
    log_det = 2.0 * np.sum(np.log(np.diag(chol[0])))
    # norm is DERIVED from scipy's normalization 1/((2*pi)^(d/2) * sqrt(det(H))), not tuned:
    #   sqrt(det(H)) = exp(0.5*log_det)  =>  norm = exp(-0.5*(log_det + d*log(2*pi))).
    norm = np.exp(-0.5 * (log_det + d * np.log(2.0 * np.pi)))

    m = grid_points.shape[1]
    out = np.zeros(m, dtype=np.float64)
    k = data.shape[1]
    for start in range(0, k, train_block):
        sl = slice(start, min(start + train_block, k))
        diff = grid_points[:, None, :] - data[:, sl, None]  # (2, kb, m)
        kb = diff.shape[1]
        flat = diff.reshape(d, kb * m)
        tdiff = cho_solve(chol, flat)  # whiten (2, kb*m)
        energy = 0.5 * np.sum(flat * tdiff, axis=0).reshape(kb, m)  # (kb, m)
        out += np.einsum("k,km->m", w[sl], np.exp(-energy))
    out *= norm
    return out.reshape(GRID_NX, GRID_NY)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------


class GhostGkModel:
    """League-average GK positioning model using RFCDE density estimation.

    Uses HistGradientBoostingRegressor leaf-assignment (NaN-tolerant)
    + weighted 2D KDE for joint density estimation. The regressor is
    trained on gk_x as a proxy target — its role is purely to partition
    feature space into regions of similar game state.

    Thread-safe for concurrent predict/predict_density calls after fit/load.

    Examples
    --------
    >>> model = GhostGkModel()
    >>> model.fit(X_train, labels_train)
    >>> positions = model.predict(X_test)  # shape (n, 2)
    >>> densities = model.predict_density(X_test)  # list[GhostGkDensity]

    See NOTICE for full bibliographic citations.
    """

    def __init__(self, *, n_estimators: int = 500, max_depth: int = 8, verbose: int = 0):
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._verbose = verbose
        self._tree_nodes: list[np.ndarray] | None = None
        self._training_gk_x: np.ndarray | None = None
        self._training_gk_y: np.ndarray | None = None
        self._training_leaves: np.ndarray | None = None
        # Transient sklearn regressors — available after fit(), not after load()
        self._regressor_x: object | None = None
        self._regressor_y: object | None = None

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame) -> GhostGkModel:
        """Train on feature matrix + (gk_x, gk_y) labels.

        Parameters
        ----------
        features : pd.DataFrame
            Shape (n_samples, 26).
        labels : pd.DataFrame
            Columns "gk_x", "gk_y" — actual GK positions (goal-relative).

        Examples
        --------
        >>> model = GhostGkModel()
        >>> model.fit(X_train, labels_train)
        """
        from sklearn.ensemble import HistGradientBoostingRegressor

        X = features.values.astype(np.float64)
        y_x = labels["gk_x"].values.astype(np.float64)

        # Determine categorical feature index (phase)
        phase_idx = list(features.columns).index("phase") if "phase" in features.columns else None
        cat_features = [phase_idx] if phase_idx is not None else []

        cat_arg: list[int] | str = cat_features if cat_features else "from_dtype"
        regressor = HistGradientBoostingRegressor(
            max_iter=self._n_estimators,
            max_depth=self._max_depth,
            categorical_features=cat_arg,  # type: ignore[arg-type]
            random_state=42,
            verbose=self._verbose,
        )
        regressor.fit(X, y_x)

        # Train gk_y regressor (same hyperparams) for fast predict_mean
        y_y = labels["gk_y"].values.astype(np.float64)
        regressor_y = HistGradientBoostingRegressor(
            max_iter=self._n_estimators,
            max_depth=self._max_depth,
            categorical_features=cat_arg,  # type: ignore[arg-type]
            random_state=42,
            verbose=self._verbose,
        )
        regressor_y.fit(X, y_y)

        # Keep sklearn regressors for fast predict_mean (transient, not serialized)
        self._regressor_x = regressor
        self._regressor_y = regressor_y

        # Extract tree node arrays for serialization + inference (gk_x trees only)
        self._tree_nodes = []
        for tree_list in regressor._predictors:
            tree = tree_list[0]
            self._tree_nodes.append(tree.nodes.copy())

        # Compute training leaves
        self._training_leaves = _vectorized_leaf_indices(self._tree_nodes, X)
        self._training_gk_x = np.array(y_x, copy=True)
        self._training_gk_y = np.asarray(y_y, dtype=np.float64).copy()

        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict joint (x, y) mode for each sample.

        Returns shape (n_samples, 2) — argmax of joint 2D density grid.

        Examples
        --------
        >>> model.predict(X_test).shape
        (100, 2)
        """
        densities = self.predict_density(features)
        return np.array([[d.mode_x, d.mode_y] for d in densities])

    def predict_mean(self, features: pd.DataFrame) -> np.ndarray:
        """Fast point prediction using the underlying regressors.

        Uses sklearn's Cython-optimized predict for both gk_x and gk_y.
        Only available after `fit()` (not after `load()`, which discards
        the sklearn regressors). Orders of magnitude faster than
        `predict` (which runs per-sample KDE on a 60x64 grid) — suitable
        for CV evaluation and batch scoring.

        Returns shape (n_samples, 2) — predicted (x, y).

        Examples
        --------
        >>> model.predict_mean(X_test).shape
        (100, 2)
        """
        if self._regressor_x is None or self._regressor_y is None:
            msg = (
                "predict_mean requires sklearn regressors (available after "
                "fit(), not after load()). Use predict() instead."
            )
            raise RuntimeError(msg)

        from sklearn.ensemble import HistGradientBoostingRegressor

        reg_x: HistGradientBoostingRegressor = self._regressor_x  # type: ignore[assignment]
        reg_y: HistGradientBoostingRegressor = self._regressor_y  # type: ignore[assignment]

        X = features.values.astype(np.float64)
        result = np.empty((len(X), 2), dtype=np.float64)
        result[:, 0] = reg_x.predict(X)
        result[:, 1] = reg_y.predict(X)
        return result

    def predict_density(
        self, features: pd.DataFrame, *, kde_backend: str = "vectorized"
    ) -> list[GhostGkDensity]:
        """Full density prediction per sample.

        Computes leaf co-occurrence weights, weighted 2D KDE, grid evaluation.

        Examples
        --------
        >>> densities = model.predict_density(X_test)
        >>> densities[0].probabilities.sum()
        1.0
        """
        if (
            self._tree_nodes is None
            or self._training_leaves is None
            or self._training_gk_x is None
            or self._training_gk_y is None
        ):
            msg = "Model not fitted. Call .fit() or .load() first."
            raise RuntimeError(msg)

        training_gk_x = self._training_gk_x
        training_gk_y = self._training_gk_y

        X = features.values.astype(np.float64)
        query_leaves = _vectorized_leaf_indices(self._tree_nodes, X)

        # Precompute grid mesh
        grid_xx, grid_yy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
        grid_points = np.vstack([grid_xx.ravel(), grid_yy.ravel()])

        # Vectorized leaf-match weights for all query samples at once (block-streamed).
        all_weights = _leaf_match_weights(self._training_leaves, query_leaves)

        results: list[GhostGkDensity] = []
        for i in range(len(X)):
            weights = all_weights[i]

            nonzero = weights > 0
            w = weights[nonzero]
            gk_x_w = training_gk_x[nonzero]
            gk_y_w = training_gk_y[nonzero]

            if len(w) < 2:
                probs = np.ones((GRID_NX, GRID_NY)) / (GRID_NX * GRID_NY)
            else:
                try:
                    if kde_backend == "scipy":
                        probs = _kde_density_scipy(gk_x_w, gk_y_w, w, grid_points)
                    elif kde_backend == "vectorized":
                        probs = _kde_density_vectorized(gk_x_w, gk_y_w, w, grid_points)
                    else:
                        msg = f"Unknown kde_backend: {kde_backend!r}"
                        raise ValueError(msg)
                    total = probs.sum()
                    if total > 0:
                        probs = probs / total
                    else:
                        probs = np.ones((GRID_NX, GRID_NY)) / (GRID_NX * GRID_NY)
                except np.linalg.LinAlgError:
                    probs = np.ones((GRID_NX, GRID_NY)) / (GRID_NX * GRID_NY)

            # Joint 2D mode
            flat_idx = int(np.argmax(probs))
            ix, iy = np.unravel_index(flat_idx, probs.shape)
            mode_x = float(_GRID_X[ix])
            mode_y = float(_GRID_Y[iy])

            # Mean
            mean_x = float(np.sum(probs * grid_xx))
            mean_y = float(np.sum(probs * grid_yy))

            # Spread (entropy)
            nz = probs[probs > 0]
            entropy = float(-np.sum(nz * np.log(nz)))
            spread = float(np.exp(entropy) * GRID_RESOLUTION**2)

            results.append(
                GhostGkDensity(
                    mode_x=mode_x,
                    mode_y=mode_y,
                    mean_x=mean_x,
                    mean_y=mean_y,
                    spread=spread,
                    grid_x=_GRID_X,
                    grid_y=_GRID_Y,
                    probabilities=probs,
                )
            )

        return results

    def save(self, path: Path) -> None:
        """Serialize to npz + metadata.json + SHA256SUMS (no pickle).

        Artifact structure:
        - rfcde_weights.npz: tree_nodes_* arrays + training_gk_x/y + training_leaves
        - metadata.json: feature_names, grid_spec, hyperparams, version
        - SHA256SUMS: per-file integrity hashes

        Examples
        --------
        >>> model.save(Path("models/ghost_gk_v1"))
        """
        if (
            self._tree_nodes is None
            or self._training_gk_x is None
            or self._training_gk_y is None
            or self._training_leaves is None
        ):
            msg = "Model not fitted. Call .fit() first."
            raise RuntimeError(msg)

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 1. Weights npz
        save_dict: dict[str, np.ndarray] = {
            "training_gk_x": self._training_gk_x,
            "training_gk_y": self._training_gk_y,
            "training_leaves": self._training_leaves,
            "n_trees": np.array([len(self._tree_nodes)]),
        }
        for i, nodes in enumerate(self._tree_nodes):
            save_dict[f"tree_nodes_{i}"] = nodes.view(np.uint8)
            save_dict[f"tree_dtype_{i}"] = np.array([str(nodes.dtype)], dtype="U2000")

        npz_path = path / "rfcde_weights.npz"
        np.savez_compressed(str(npz_path), **save_dict)  # type: ignore[arg-type]

        # 2. Metadata
        metadata = {
            "feature_names": GHOST_GK_FEATURE_NAMES,
            "grid_spec": {
                "x_min": GRID_X_MIN,
                "x_max": GRID_X_MAX,
                "y_min": GRID_Y_MIN,
                "y_max": GRID_Y_MAX,
                "nx": GRID_NX,
                "ny": GRID_NY,
                "resolution": GRID_RESOLUTION,
            },
            "n_estimators": self._n_estimators,
            "max_depth": self._max_depth,
            "version": "1.0.0",
        }
        meta_path = path / "metadata.json"
        with open(meta_path, "w", newline="\n") as f:
            json.dump(metadata, f, indent=2)

        # 3. SHA-256
        sums_path = path / "SHA256SUMS"
        with open(sums_path, "w", newline="\n") as f:
            for fname in ["rfcde_weights.npz", "metadata.json"]:
                raw = (path / fname).read_bytes()
                # Normalize CRLF→LF for text files so hash is platform-independent
                if fname.endswith(".json"):
                    raw = raw.replace(b"\r\n", b"\n")
                h = hashlib.sha256(raw).hexdigest()
                f.write(f"{h}  {fname}\n")

    @classmethod
    def load(cls, path: Path) -> GhostGkModel:
        """Load from local directory with SHA-256 verification.

        No sklearn or onnxruntime needed — tree traversal uses stored
        node arrays with numpy.

        Examples
        --------
        >>> model = GhostGkModel.load(Path("models/ghost_gk_v1"))
        """
        path = Path(path)

        # Verify integrity
        sums_path = path / "SHA256SUMS"
        if not sums_path.exists():
            raise IntegrityError(f"SHA256SUMS not found in {path}")

        with open(sums_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                expected_hash, fname = line.split("  ", 1)
                raw = (path / fname).read_bytes()
                # Normalize CRLF→LF for text files so hash is platform-independent
                if fname.endswith(".json"):
                    raw = raw.replace(b"\r\n", b"\n")
                actual_hash = hashlib.sha256(raw).hexdigest()
                if actual_hash != expected_hash:
                    raise IntegrityError(
                        f"Integrity check failed for {fname}: expected {expected_hash}, got {actual_hash}"
                    )

        # Load metadata
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        # Load npz (use context manager to release file handle on Windows)
        with np.load(path / "rfcde_weights.npz", allow_pickle=False) as data:
            n_trees = int(data["n_trees"][0])
            tree_nodes = []
            for i in range(n_trees):
                raw_bytes = np.array(data[f"tree_nodes_{i}"])
                dtype_str = str(data[f"tree_dtype_{i}"][0])
                # ast.literal_eval safely parses the dtype descriptor (list of tuples)
                dtype = np.dtype(ast.literal_eval(dtype_str))
                nodes = raw_bytes.view(dtype)
                tree_nodes.append(nodes)

            training_gk_x = np.array(data["training_gk_x"])
            training_gk_y = np.array(data["training_gk_y"])
            training_leaves = np.array(data["training_leaves"])

        model = cls(
            n_estimators=metadata.get("n_estimators", 500),
            max_depth=metadata.get("max_depth", 8),
        )
        model._tree_nodes = tree_nodes
        model._training_gk_x = training_gk_x
        model._training_gk_y = training_gk_y
        model._training_leaves = training_leaves

        return model

    @classmethod
    def from_variant(cls, variant: GhostGkVariant = "default") -> GhostGkModel:
        """Load a model variant by name.

        ``"default"`` is bundled in the wheel. ``"full"`` is downloaded
        from HuggingFace Hub on first use (requires ``[ghost-gk]`` extra).

        Parameters
        ----------
        variant : "default" | "full"
            ``"default"``: lightweight model (~9 MB, 36 k training samples).
            ``"full"``: high-resolution model (~91 MB, 537 k training samples).

        Examples
        --------
        >>> model = GhostGkModel.from_variant("full")
        """
        weights_dir = _WEIGHTS_ROOT / variant
        if (weights_dir / "SHA256SUMS").exists():
            return cls.load(weights_dir)
        if variant == "full":
            return cls.from_hub(_HF_REPO_ID)
        msg = f"Bundled Ghost-GK weights not found at {weights_dir}"
        raise FileNotFoundError(msg)

    @classmethod
    def from_hub(cls, repo_id: str = _HF_REPO_ID) -> GhostGkModel:
        """Download from HuggingFace Hub and load.

        Requires ``pip install silly-kicks[ghost-gk]``.

        Examples
        --------
        >>> model = GhostGkModel.from_hub()
        """
        try:
            from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
        except ImportError:
            msg = "Ghost GK full model requires: pip install silly-kicks[ghost-gk]"
            raise ImportError(msg) from None

        local_dir = snapshot_download(repo_id=repo_id)
        return cls.load(Path(local_dir))


# ---------------------------------------------------------------------------
# Per-frame primitive
# ---------------------------------------------------------------------------


def compute_ghost_gk(
    frames: pd.DataFrame,
    *,
    model: GhostGkModel | GhostGkVariant | None = None,
    home_team_id: int | str,
    actions: pd.DataFrame | None = None,
    link_frame_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Per-frame ghost-GK primitive (batched).

    Adds ghost_gk_x, ghost_gk_y, ghost_gk_spread columns.
    One prediction per (frame, GK team). Results written to GK rows.

    Input frames MUST be in LTR-normalized convention (home team attacks
    right in all periods --- standard silly-kicks tracking output).

    Parameters
    ----------
    frames : pd.DataFrame
        Tracking frames (TRACKING_FRAMES_COLUMNS schema, LTR-normalized).
    model : GhostGkModel | "default" | "full" | None
        ``"default"`` / ``None``: bundled lightweight model (~9 MB).
        ``"full"``: high-resolution bundled model (~91 MB).
        Or a pre-loaded ``GhostGkModel`` instance.
    home_team_id : int | str
        Home team ID (attacks right -> defends at x=0).
    actions : pd.DataFrame | None
        SPADL actions for score_diff and phase context. If None, both
        default to 0 (backward-compatible with 3.19.0 behaviour).
    link_frame_ids : set[int] | None, default None
        When provided, restrict the per-sample KDE (``predict_density``) to GK
        samples whose ``frame_id`` is in this set. Feature extraction still runs
        over the FULL frames, so the per-period defending-goal mean-x and the
        cross-period one-step velocity state are preserved exactly --- the KDE is
        per-sample independent, so the restricted result is byte-identical to the
        unrestricted one for the kept frames. When None, every sample is predicted
        (backward-compatible). See PR-S66 spec sections 2-3.

    Returns
    -------
    pd.DataFrame
        Copy of frames with ghost_gk_x, ghost_gk_y, ghost_gk_spread added.

    Examples
    --------
    >>> from silly_kicks.tracking._ghost_gk import compute_ghost_gk
    >>> result = compute_ghost_gk(frames, home_team_id=1)
    """
    resolved = _resolve_model(model)
    out = frames.copy()
    out["ghost_gk_x"] = np.nan
    out["ghost_gk_y"] = np.nan
    out["ghost_gk_spread"] = np.nan

    # Build context callbacks from actions
    score_fn = _build_score_lookup(actions, home_team_id) if actions is not None else None
    phase_fn = _build_phase_lookup(actions) if actions is not None else None

    # PR-S66 §5: restrict BOTH the heavy feature extraction and the per-sample KDE
    # to the action-linked frames. _extract_all_ghost_gk_features still walks every
    # frame to maintain the cross-period one-step velocity state and computes the
    # per-period defending-goal mean-x over the full frames, so the linked-frame
    # features --- and the per-sample KDE, which has zero cross-sample coupling ---
    # are byte-identical to the unrestricted compute. See TestExtractionRestriction.
    batch_features, meta = _extract_all_ghost_gk_features(
        frames,
        home_team_id=home_team_id,
        score_at_time=score_fn,
        phase_at_time=phase_fn,
        link_frame_ids=link_frame_ids,
    )

    if len(batch_features) == 0:
        return out

    # Batch predict
    densities = resolved.predict_density(batch_features)

    # Build result DataFrame from predictions (single merge, not O(n*m) loop)
    result_df = pd.DataFrame(
        {
            "game_id": meta["game_id"].values,
            "period_id": meta["period_id"].values,
            "frame_id": meta["frame_id"].values,
            "team_id": meta["gk_team_id"].values,
            "ghost_gk_x": [d.mode_x for d in densities],
            "ghost_gk_y": [d.mode_y for d in densities],
            "ghost_gk_spread": [d.spread for d in densities],
        }
    )

    # Merge into GK rows via single join
    gk_mask = out["is_goalkeeper"].astype(bool) & ~out["is_ball"].astype(bool)
    gk_rows_df = out.loc[gk_mask, ["game_id", "period_id", "frame_id", "team_id"]].copy()
    gk_rows_df = gk_rows_df.merge(
        result_df,
        on=["game_id", "period_id", "frame_id", "team_id"],
        how="left",
    )
    out.loc[gk_mask, "ghost_gk_x"] = gk_rows_df["ghost_gk_x"].values
    out.loc[gk_mask, "ghost_gk_y"] = gk_rows_df["ghost_gk_y"].values
    out.loc[gk_mask, "ghost_gk_spread"] = gk_rows_df["ghost_gk_spread"].values

    return out
