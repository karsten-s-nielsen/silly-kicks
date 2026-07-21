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
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import gaussian_kde

from silly_kicks.id_compat import ids_match, same_id
from silly_kicks.spadl import config as spadlconfig

from ._ball_carrier import DEFAULT_CARRIER_PARAMS, infer_ball_carrier

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
# Warning categories
# ---------------------------------------------------------------------------


class GhostClampWarning(UserWarning):
    """The served ghost position fell outside the physical pitch and was clamped.

    A dedicated category so a consumer can silence the batch-clamp notice without
    silencing every ``UserWarning`` from ``tracking`` --- it is emitted from two public
    entry points (``compute_ghost_gk`` and ``serve_ghost_gk_positions``).

    A clamp is a signal about the INPUT, not a routine rounding step: the served position
    left the physical pitch, which in practice means the model extrapolated outside its
    trained label hull (ADR-016) after something upstream mis-flagged ``is_goalkeeper`` and
    wrong-footed the goal-side flip. Escalating it is the cheap way to catch that.

    Examples
    --------
    Make a clamp fatal in a batch job, so bad orientation fails loudly instead of
    silently serving a pitch-edge position::

        import warnings
        from silly_kicks.tracking import GhostClampWarning, compute_ghost_gk

        warnings.filterwarnings("error", category=GhostClampWarning)
        ghost = compute_ghost_gk(frames, home_team_id=home_id)
    """


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

#: Name of the served point estimate, recorded in metadata.json (R3-style provenance)
#: and asserted on load(). "boosted_mean" = the exact sklearn HGBR boosted prediction
#: reconstructed pickle-free (spec 2026-06-04 Option A §3.1); load() fails closed on a
#: conflicting tag (train/serve skew guard).
SERVED_ESTIMATOR = "boosted_mean"


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
        - ``None`` or ``"default"``: lightweight model (~12 MB, 36 k samples).
          Bundled in the wheel — works offline, no download needed.
        - ``"full"``: high-resolution model (~170 MB, 887 k samples).
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

# Which providers' feeds carry a per-player detection flag (spec 4.3).
# A null `visibility` is AMBIGUOUS: for a fully-observed provider it means "no flag exists and
# none is needed"; for a detection-aware provider it means "the pipeline DISCARDED the flag"
# (the kloppy gateway hard-codes visibility=None). Reading the second as the first would train
# ghost-GK on interpolator output -- ~80% of SkillCorner keeper positions are extrapolated.
_DETECTION_AWARE_PROVIDERS = frozenset({"skillcorner"})
# metrica is full optical tracking (all players every frame, NO detection flag) -- fully observed
# like the native providers. Classifying it here keeps ghost-GK trainable on metrica (a pre-PR
# capability the always-run detected-only filter would otherwise crash); metrica's exclusion from
# the registered GKDV corpora is a separate corpus-composition decision (Tier-2 data quality).
_FULLY_OBSERVED_PROVIDERS = frozenset({"gradientsports", "sportec", "idsse", "metrica"})


def keeper_detection_mask(visibility: pd.Series, *, provider: str) -> np.ndarray:
    """Rows whose keeper was ACTUALLY DETECTED. Fail-closed on the ambiguous null (spec 4.3)."""
    if provider in _FULLY_OBSERVED_PROVIDERS:
        return np.ones(len(visibility), dtype=bool)
    if provider not in _DETECTION_AWARE_PROVIDERS:
        raise ValueError(
            f"keeper_detection_mask: unknown provider {provider!r}. Add it to "
            "_DETECTION_AWARE_PROVIDERS or _FULLY_OBSERVED_PROVIDERS -- an unknown provider is "
            "NOT assumed observed."
        )
    if visibility.isna().all():
        raise ValueError(
            f"keeper_detection_mask: provider {provider!r} carries a detection flag, but "
            "`visibility` is entirely null -- the pipeline discarded it (the kloppy gateway "
            "hard-codes visibility=None). Build these frames with tracking.skillcorner instead; "
            "training on undetected keepers means training on the interpolator (spec 4.3)."
        )
    return visibility.fillna(False).astype(bool).to_numpy()


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

    # Build per-game cumulative score arrays.
    #
    # `is_home` MUST route through the ADR-019 seam, not a naive `str(t) == str(home_team_id)`:
    # the latter renders a float-backed id as "366.0" against a scalar "366", so EVERY goal
    # falls to the away side. Measured on a 3-goal fixture (2 home, 1 away) with a float64
    # `team_id`: score_diff came back -3 instead of +1. That feeds `score_diff`, one of the 26
    # ghost-GK features, so the error is a four-goal swing on a trained-model input rather than
    # a rounding nuisance. `ids_match` is vectorized, so this is also cheaper than the
    # per-element Python loop it replaces.
    _lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for gid, grp in goals.groupby("game_id"):
        times = np.asarray(grp["time_seconds"].values, dtype=np.float64)
        is_home = ids_match(grp["_scoring_team"], home_team_id).to_numpy()
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
    gk_team_id: float | int | str,
    goal_x: float = 0.0,
    score_diff: float = 0.0,
    phase: int = 0,
    ball_carrier_team_id: float | int | str | None = None,
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
    gk_team_id : float | int | str
        Team ID of the GK whose ghost position we predict. ``float`` is admitted
        deliberately, not as a convenience: a float-backed team id is what a merge or an
        NaN-carrying column leaves behind (``infer_ball_carrier`` emits boxed floats, and
        the zero-row serve path types ``gk_team_id`` float64), and ``ids_match`` resolves
        it via ``canonical_id`` -- ``canonical_id(366.0) == canonical_id("366")``. Typing
        it away would have made the ONE dtype this function must survive unrepresentable.
    goal_x : float
        x-coordinate of the defending goal (0.0 or 105.0).
    score_diff : float
        GK's team score minus opponent.
    phase : int
        0 = open_play, 1 = set_piece, 2 = goal_kick.
    ball_carrier_team_id : float | int | str | None
        Team currently in possession. Same id-dtype latitude as ``gk_team_id`` above --
        this is precisely the column ``infer_ball_carrier`` hands over as boxed floats.
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
    # ADR-019: gk_team_id is a caller-supplied scalar and need NOT share the frames'
    # team_id dtype (GS emits nullable Int64; sportec/kloppy carry object strings). A raw
    # compare is silently all-False across dtypes -> an EMPTY defending split and a corrupt
    # feature row rather than an error.
    #
    # The attacking split is `~ids_match`, NOT `ids_differ`: ids_differ requires BOTH sides
    # present, so it would move an NA-team player out of `attacking` (where the original
    # `!=` put it) into neither split. `~ids_match` reproduces the original NA semantics
    # exactly -- for an NA player team_id AND for an NA gk_team_id.
    _is_gk_team = ids_match(players["team_id"], gk_team_id)
    _is_gk_flag = players["is_goalkeeper"].astype(bool)
    defending = players[_is_gk_team & ~_is_gk_flag]
    attacking = players[~_is_gk_team & ~_is_gk_flag]
    gk_rows = players[_is_gk_team & _is_gk_flag]

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
    # ADR-019 scalar-vs-scalar seam. `ball_carrier_team_id` may reach here from the public
    # `carrier=` cache kwarg with a dtype the frames do not share, in which case a raw
    # compare pins team_in_possession to 0 for every row. `same_id` also subsumes the
    # try/except this replaced: it returns False for None/pd.NA/NaN on either side rather
    # than raising (the fence's stated reason -- "pd.NA comparison raises TypeError" -- is
    # handled inside `_canonical`), so the None/NA/NaN behaviour is unchanged.
    team_in_poss = 1.0 if same_id(ball_carrier_team_id, gk_team_id) else 0.0
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
        (n_samples, 8): game_id, period_id, frame_id, gk_team_id,
        gk_x_gr, gk_y_gr, gk_player_id, gk_visibility. The last two carry
        the keeper's identity and per-frame detection flag off the SAME
        frames row the label came from (spec 4.3: keeper-grouped CV +
        detected-only targets).

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
            goal_x = _defending_goal.get((gid, pid, gk_team), 0.0 if same_id(gk_team, home_team_id) else _FIELD_LENGTH)
            flip = goal_x > 50.0

            # Cheap defensive-line-x + centroid in goal-relative coords, computed
            # for EVERY frame to drive the velocity state. These mirror exactly
            # extract_ghost_gk_features' defensive_line_x and the stored centroid
            # (median of the back-4 goal-relative x; mean goal-relative x) — see
            # TestExtractionRestriction golden which guards bit-identical velocity.
            # ADR-019: must use the SAME id-identity rule as extract_ghost_gk_features'
            # defending split -- the TestExtractionRestriction golden guards that the
            # velocity state computed here is bit-identical to the extractor's.
            defending = frame_data[
                ids_match(frame_data["team_id"], gk_team)
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
                    if not same_id(gk_team, home_team_id):
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
                        "gk_player_id": gk_row["player_id"],
                        "gk_visibility": gk_row.get("visibility"),
                    }
                )

            # Update velocity state (every frame, linked or not)
            prev_state[state_key] = (dl_x, def_cx)
            prev_timestamps[state_key] = time_s

    if not feature_rows:
        return (
            pd.DataFrame(columns=GHOST_GK_FEATURE_NAMES),
            pd.DataFrame(
                columns=[
                    "game_id",
                    "period_id",
                    "frame_id",
                    "gk_team_id",
                    "gk_x_gr",
                    "gk_y_gr",
                    "gk_player_id",
                    "gk_visibility",
                ]
            ),
        )

    features = pd.concat(feature_rows, ignore_index=True)
    meta = pd.DataFrame(meta_rows)
    return features, meta


@overload
def prepare_ghost_gk_training_data(
    frames: pd.DataFrame,
    *,
    home_team_id: str | int,
    actions: pd.DataFrame | None = ...,
    subsample_fps: float | None = ...,
    carrier_params: dict | None = ...,
    return_meta: Literal[False] = ...,
) -> tuple[pd.DataFrame, pd.DataFrame]: ...


@overload
def prepare_ghost_gk_training_data(
    frames: pd.DataFrame,
    *,
    home_team_id: str | int,
    actions: pd.DataFrame | None = ...,
    subsample_fps: float | None = ...,
    carrier_params: dict | None = ...,
    return_meta: Literal[True],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: ...


def prepare_ghost_gk_training_data(
    frames: pd.DataFrame,
    *,
    home_team_id: str | int,
    actions: pd.DataFrame | None = None,
    subsample_fps: float | None = 1.0,
    carrier_params: dict | None = None,
    return_meta: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    carrier_params : dict | None
        Ball-carrier scoring params (``tolerance_m``/``beta``/``gamma``) used to
        compute the ``team_in_possession`` feature. ``None`` uses the library
        default (:data:`DEFAULT_CARRIER_PARAMS`); the trainer passes the same dict
        here and to :meth:`GhostGkModel.fit` so recorded == used (R3, PR-S81).
    return_meta : bool, default False
        When True, also return the per-row ``meta`` frame (spec 4.3) so callers
        can build keeper-grouped CV (``gk_player_id``) and enforce detected-only
        targets (``gk_visibility`` + :func:`keeper_detection_mask`). ``meta`` is
        filtered by the SAME masks as ``features``/``labels`` so its rows stay
        aligned. Default False keeps the documented ``(features, labels)`` shape
        the four existing call sites depend on.

    Returns
    -------
    features : pd.DataFrame
        (n_samples, len(GHOST_GK_FEATURE_NAMES)) with GHOST_GK_FEATURE_NAMES
        columns.
    labels : pd.DataFrame
        (n_samples, 2) with columns "gk_x", "gk_y" in goal-relative
        coordinates matching the GhostGkModel training domain
        ([0, 30] x [18, 50]).
    meta : pd.DataFrame, optional
        Only when ``return_meta=True``. Row-aligned with ``features``/``labels``;
        carries ``game_id, period_id, frame_id, gk_team_id, gk_x_gr, gk_y_gr,
        gk_player_id, gk_visibility``.

    Examples
    --------
    >>> features, labels = prepare_ghost_gk_training_data(
    ...     frames, home_team_id=1, actions=actions, subsample_fps=1.0
    ... )
    >>> model = GhostGkModel()
    >>> model.fit(features, labels)
    """
    import warnings

    # Build context callbacks
    score_fn = _build_score_lookup(actions, home_team_id) if actions is not None else None
    phase_fn = _build_phase_lookup(actions) if actions is not None else None

    # Carrier (always computed --- only needs frames). carrier_params=None is
    # byte-identical to the historical bare call because DEFAULT_CARRIER_PARAMS
    # equals infer_ball_carrier's signature defaults (PR-S81 / R3 single source).
    cp: dict = dict(carrier_params) if carrier_params else dict(DEFAULT_CARRIER_PARAMS)
    carrier_raw = infer_ball_carrier(frames, **cp)
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
        empty_features = pd.DataFrame(columns=GHOST_GK_FEATURE_NAMES)
        empty_labels = pd.DataFrame(columns=["gk_x", "gk_y"])
        # `meta` here is the 8-column empty frame from _extract_all_ghost_gk_features.
        if return_meta:
            return empty_features, empty_labels, meta
        return empty_features, empty_labels

    # Extract labels
    labels = meta[["gk_x_gr", "gk_y_gr"]].rename(columns={"gk_x_gr": "gk_x", "gk_y_gr": "gk_y"})

    # Drop NaN labels (GK not visible). meta MUST be filtered by the same mask so keeper
    # identity (gk_player_id) stays aligned with its row (spec 4.3, Known-risks #2).
    valid = labels["gk_x"].notna() & labels["gk_y"].notna()
    features = features[valid.values].reset_index(drop=True)
    labels = labels[valid.values].reset_index(drop=True)
    meta = meta[valid.values].reset_index(drop=True)

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
        meta = meta[in_domain.values].reset_index(drop=True)

    if return_meta:
        return features, labels, meta
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


def _vectorized_leaf_values(nodes_list: list[np.ndarray], X: np.ndarray) -> np.ndarray:
    """Sum of the reached-leaf ``value`` across all trees — the HGBR raw additive prediction.

    Sibling of :func:`_vectorized_leaf_indices`: identical traversal, but instead of
    returning the reached leaf *index* per tree it accumulates the reached leaf's
    ``value`` field. The HGBR boosted prediction (squared-error loss, identity link)
    is ``baseline + sum_trees leaf_value`` with the learning rate baked into the
    stored leaf ``value``s, so ``baseline + _vectorized_leaf_values(...)`` reconstructs
    ``HistGradientBoostingRegressor.predict`` exactly (parity gate ≤ 1e-6).

    Numeric thresholds only — valid because :meth:`GhostGkModel.fit` trains with no
    categorical features (``categorical_features=None``; spec §3.2), so every split is
    a ``num_threshold`` comparison and no ``raw_left_cat_bitsets`` routing is needed.
    Same NaN / ``missing_go_to_left`` handling as :func:`_vectorized_leaf_indices`.

    Parameters
    ----------
    nodes_list : list[np.ndarray]
        One structured node array per tree (fields: ``left``, ``right``,
        ``feature_idx``, ``num_threshold``, ``missing_go_to_left``, ``value``).
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features); fit-time column order.

    Returns
    -------
    np.ndarray
        Shape (n_samples,) — summed leaf value per sample.

    Examples
    --------
    >>> # baseline + _vectorized_leaf_values(trees, X) == regressor.predict(X)
    >>> # (see tests/tracking/test_ghost_gk_serve_mean.py parity gate)
    """
    n_samples = X.shape[0]
    total = np.zeros(n_samples, dtype=np.float64)

    for nodes in nodes_list:
        current = np.zeros(n_samples, dtype=np.intp)

        for _ in range(100):  # depth bound
            node_data = nodes[current]
            is_leaf = node_data["left"] == 0
            if np.all(is_leaf):
                break

            feat_vals = X[np.arange(n_samples), node_data["feature_idx"]]
            go_left = np.where(
                np.isnan(feat_vals),
                node_data["missing_go_to_left"].astype(bool),
                feat_vals <= node_data["num_threshold"],
            )
            next_node = np.where(go_left, node_data["left"], node_data["right"])
            current = np.where(is_leaf, current, next_node)

        # Convergence guard (raise, NOT assert — `python -O` strips asserts): a tree
        # deeper than the cap would otherwise read an internal node's `value` (garbage).
        if not np.all(nodes[current]["left"] == 0):
            msg = "leaf traversal did not converge within depth cap"
            raise RuntimeError(msg)
        total += nodes[current]["value"]

    return total


def _leaf_match_weights(
    training_leaves: np.ndarray,
    query_leaves: np.ndarray,
    *,
    query_block: int = 64,
) -> np.ndarray:
    """Per-query leaf-match weight = fraction of trees with matching leaf.

    training_leaves : (n_train, n_trees); query_leaves : (n_query, n_trees).
    Returns (n_query, n_train) weights. Streams queries in blocks to bound the
    (qb, n_train, n_trees) broadcast for the 887k-train variant.
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


def _kde_setup(
    gk_x_w: np.ndarray,
    gk_y_w: np.ndarray,
    w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, float]:
    """Shared closed-form KDE setup for the numpy + numba kernels (Leg-B: identical setup).

    Weighted Scott-bandwidth covariance + Cholesky PD-branch + self-consistent det/norm.
    cho_factor raises np.linalg.LinAlgError on a non-PD covariance (collinear/identical points)
    -- same as scipy's cholesky -- so predict_density's ``except np.linalg.LinAlgError`` degrades
    to the uniform grid (the singular boundary is byte-identical to 4.2.0). ``norm`` is DERIVED
    from scipy's normalization 1/((2*pi)^(d/2) * sqrt(det(H))): sqrt(det(H)) = exp(0.5*log_det),
    so norm = exp(-0.5*(log_det + d*log(2*pi))). ``det`` is derived from the SAME factor as
    log_det/norm (det(H) = det(L)^2 = (L00*L11)^2, chol[0] holding L on its diagonal for
    lower=True), keeping the whitening (1/det) and normalization self-consistent in the
    near-singular zone and avoiding the catastrophic cancellation of an independently-rounded
    h11*h22 - h12^2 as det -> 0.

    Returns (data, w, h11, h12, h22, det, norm): data (2, k) float64, w normalized.
    """
    from scipy.linalg import cho_factor

    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    data = np.vstack([np.asarray(gk_x_w, np.float64), np.asarray(gk_y_w, np.float64)])  # (2, k)
    d = 2
    neff = 1.0 / np.sum(w**2)
    factor = neff ** (-1.0 / (d + 4))
    covariance = np.atleast_2d(np.cov(data, rowvar=True, bias=False, aweights=w)) * factor**2
    chol = cho_factor(covariance, lower=True)
    log_det = 2.0 * np.sum(np.log(np.diag(chol[0])))
    norm = np.exp(-0.5 * (log_det + d * np.log(2.0 * np.pi)))
    det = (chol[0][0, 0] * chol[0][1, 1]) ** 2
    return data, w, covariance[0, 0], covariance[0, 1], covariance[1, 1], det, norm


def _kde_density_vectorized(
    gk_x_w: np.ndarray,
    gk_y_w: np.ndarray,
    w: np.ndarray,
    grid_points: np.ndarray,
    *,
    train_block: int = 1024,
) -> np.ndarray:
    """Vectorized weighted Gaussian KDE on the fixed grid (closed-form 2x2 whitening).

    Reuses ``_kde_setup`` (shared with the cpu-numba backend) for the weighted covariance +
    Cholesky PD-branch + det/norm, then streams the training subset in blocks of ``train_block``,
    computing the closed-form 2x2 Mahalanobis energy directly from dx,dy. H = covariance is 2x2
    PD (cho_factor succeeded), so H^-1 = (1/det)[[h22,-h12],[-h12,h11]] and
    ``energy = 0.5/det * (h22*dx^2 - 2*h12*dx*dy + h11*dy^2)`` -- no (2,kb,m) diff/tdiff
    temporaries. Returns the norm-scaled density grid (GRID_NX, GRID_NY).

    Default ``train_block=1024`` is the conservative serverless choice: with grid m=3840, the
    largest transient ``(kb, m)`` dx/dy/energy is ~150 MB/block, safe under the Databricks 1 GB
    ``applyInPandas`` cap. The 12 MB "default" model's per-sample nonzero leaf subsets are small
    (k often < 1024 -> chunking rarely binds), so 1024 costs ~nothing there; the local benchmark
    can raise it for the in-memory "full" model.
    """
    data, w, h11, h12, h22, det, norm = _kde_setup(gk_x_w, gk_y_w, w)
    inv_det = 1.0 / det
    gx = grid_points[0]  # (m,)
    gy = grid_points[1]  # (m,)
    m = grid_points.shape[1]
    out = np.zeros(m, dtype=np.float64)
    k = data.shape[1]
    for start in range(0, k, train_block):
        sl = slice(start, min(start + train_block, k))
        dx = gx[None, :] - data[0, sl][:, None]  # (kb, m)
        dy = gy[None, :] - data[1, sl][:, None]  # (kb, m)
        energy = 0.5 * inv_det * (h22 * dx * dx - 2.0 * h12 * dx * dy + h11 * dy * dy)  # (kb, m)
        out += np.einsum("k,km->m", w[sl], np.exp(-energy))
    out *= norm
    return out.reshape(GRID_NX, GRID_NY)


def _kde_density_numba(
    gk_x_w: np.ndarray,
    gk_y_w: np.ndarray,
    w: np.ndarray,
    grid_points: np.ndarray,
) -> np.ndarray:
    """cpu-numba KDE: numpy ``_kde_setup`` (cho_factor branch + det/norm) + the @njit fused loop.

    No ``train_block`` chunking is needed -- the numba loop has no ``(k, m)`` temporaries. The
    numba module is imported lazily here so ``import silly_kicks.tracking._ghost_gk`` stays
    numba-free (numba is only required when the cpu-numba backend is actually selected).
    """
    from ._ghost_gk_numba import _kde_numba_loop

    data, w, h11, h12, h22, det, norm = _kde_setup(gk_x_w, gk_y_w, w)
    out = _kde_numba_loop(grid_points[0], grid_points[1], data[0], data[1], w, h11, h12, h22, 1.0 / det, norm)
    return out.reshape(GRID_NX, GRID_NY)


def _bin_ngp(gk_x_w: np.ndarray, gk_y_w: np.ndarray, w_norm: np.ndarray) -> np.ndarray:
    """Nearest-grid-point (NGP) binning of weighted points onto the fixed grid.

    Each point is snapped to its single nearest cell (uniform, cell-centered grid: idx =
    round((p - p0)/res)); out-of-grid points clip to the edge cell. Mass is conserved
    (field.sum() == w_norm.sum()): every point contributes its full weight to exactly one cell.
    """
    ix = np.clip(np.rint((gk_x_w - _GRID_X[0]) / GRID_RESOLUTION).astype(np.int64), 0, GRID_NX - 1)
    iy = np.clip(np.rint((gk_y_w - _GRID_Y[0]) / GRID_RESOLUTION).astype(np.int64), 0, GRID_NY - 1)
    field = np.zeros((GRID_NX, GRID_NY), dtype=np.float64)
    np.add.at(field, (ix, iy), w_norm)
    return field


def _bin_cic(gk_x_w: np.ndarray, gk_y_w: np.ndarray, w_norm: np.ndarray) -> np.ndarray:
    """Cloud-in-cell (CIC / bilinear) binning of weighted points onto the fixed grid.

    Each weighted point is spread bilinearly over its 4 surrounding cells with weights
    (1-tx)(1-ty), tx(1-ty), (1-tx)ty, tx*ty (summing to 1), instead of snapped to the single
    nearest cell (NGP). On a near-tie MULTIMODAL grid this preserves the relative peak masses, so
    the emitted mode (argmax) flips ~76% less than NGP (real data: 21/97 -> 5/97; ADR-014). Mass
    is conserved including for out-of-grid points: clip collapses indices to an edge cell but
    np.add.at still accumulates all 4 contributions, so field.sum() == w_norm.sum().
    """
    fx = (gk_x_w - _GRID_X[0]) / GRID_RESOLUTION
    fy = (gk_y_w - _GRID_Y[0]) / GRID_RESOLUTION
    i0 = np.floor(fx).astype(np.int64)
    j0 = np.floor(fy).astype(np.int64)
    tx, ty = fx - i0, fy - j0
    field = np.zeros((GRID_NX, GRID_NY), dtype=np.float64)
    for di, wx in ((0, 1.0 - tx), (1, tx)):
        ii = np.clip(i0 + di, 0, GRID_NX - 1)
        for dj, wy in ((0, 1.0 - ty), (1, ty)):
            jj = np.clip(j0 + dj, 0, GRID_NY - 1)
            np.add.at(field, (ii, jj), w_norm * wx * wy)
    return field


def _fft_convolve_field(field: np.ndarray, h11: float, h12: float, h22: float, det: float, norm: float) -> np.ndarray:
    """Shared FFT-convolution tail for the fft / fft-cic backends.

    ``field`` is the binned weighted-point grid (GRID_NX, GRID_NY). Builds the full-extent analytic
    anisotropic Gaussian kernel (identical energy form to _kde_density_vectorized) and returns the
    UNnormalized density via one zero-padded linear fftconvolve = sum_j w_j K(grid - point_j) in
    O(m log m). Binning is the SOLE per-backend difference; this tail is identical across fft /
    fft-cic (predict_density divides by .sum(), so ``norm`` cancels).
    """
    # NB: this lazy (function-scope) import is LOAD-BEARING for the k-independence spy guard
    # (test_fft*_is_k_independent_one_convolution) -- it resolves the patched scipy.signal attr at
    # call time. Do NOT hoist to module level or the spy goes blind.
    from scipy.signal import fftconvolve

    inv_det = 1.0 / det
    dx = (np.arange(-(GRID_NX - 1), GRID_NX) * GRID_RESOLUTION)[:, None]
    dy = (np.arange(-(GRID_NY - 1), GRID_NY) * GRID_RESOLUTION)[None, :]
    kernel = norm * np.exp(-0.5 * inv_det * (h22 * dx * dx - 2.0 * h12 * dx * dy + h11 * dy * dy))
    return fftconvolve(field, kernel, mode="same")


def _kde_density_fft(
    gk_x_w: np.ndarray,
    gk_y_w: np.ndarray,
    w: np.ndarray,
    grid_points: np.ndarray,  # unused: signature parity with the brute-force backends; uses the module grid
) -> np.ndarray:
    """Binned-convolution weighted Gaussian KDE on the fixed grid. O(k + m log m).

    Reuses the EXACT ``_kde_setup`` kernel as the brute-force backends (same weighted Scott
    covariance + Cholesky PD-branch + det/norm), so the convolution kernel is the identical
    anisotropic Gaussian; the SOLE approximation is snapping each training point to its nearest
    grid cell (NGP binning). Then one linear (zero-padded) FFT convolution gives
    ``sum_j w_j K(grid - point_j)`` in O(m log m), independent of the point count k -- versus the
    O(k*m) brute force (k = full training set, ~36k, on every ghost-GK prediction).

    Faithful on mean/spread always, and on the mode for UNIMODAL grids. On near-tie MULTIMODAL
    grids the NGP snap can flip which peak is the argmax, shifting the emitted mode by several
    metres (real data: up to ~6 m on ~22% of actions) -- use ``fft-cic`` (bilinear binning, ~76%
    fewer flips) or ``vectorized`` when the mode matters on multimodal distributions. NOT
    bit-faithful on the raw per-cell ``probabilities`` grid (binning quantizes per-cell mass;
    ~1.5% typical / up to ~65% on near-zero tail cells). Consumers reading the raw grid should use
    ``"vectorized"``. See ADR-014 (amended).

    ``_kde_setup`` raises ``np.linalg.LinAlgError`` on a singular covariance exactly as the other
    backends, so predict_density's uniform-fallback applies unchanged. Returns the UNnormalized
    density grid (GRID_NX, GRID_NY); predict_density divides by ``.sum()`` so ``norm`` cancels.

    Shares ``_kde_setup`` + ``_fft_convolve_field`` verbatim with ``fft-cic``; the two differ only
    in the binning step (this one ``_bin_ngp``, fft-cic ``_bin_cic``).
    """
    _data, w_n, h11, h12, h22, det, norm = _kde_setup(gk_x_w, gk_y_w, w)
    field = _bin_ngp(gk_x_w, gk_y_w, w_n)
    return _fft_convolve_field(field, h11, h12, h22, det, norm)


def _kde_density_fft_cic(
    gk_x_w: np.ndarray,
    gk_y_w: np.ndarray,
    w: np.ndarray,
    grid_points: np.ndarray,  # unused: signature parity with the brute-force backends
) -> np.ndarray:
    """Binned-convolution weighted Gaussian KDE with CIC (bilinear) binning. O(k + m log m).

    Identical to ``_kde_density_fft`` except each weighted training point is spread BILINEARLY over
    its 4 surrounding grid cells (``_bin_cic``) instead of snapped to the single nearest cell
    (``_bin_ngp``). The ``_kde_setup`` kernel build and the ``_fft_convolve_field`` convolution are
    shared verbatim. On near-tie MULTIMODAL grids CIC preserves the relative peak masses, so the
    emitted mode (argmax) flips ~76% less than NGP (real data: 21/97 -> 5/97 actions). ~2x the NGP
    bin cost, still ~1195x over ``vectorized``. Faithful on mean/spread; the raw per-cell grid is
    tighter than NGP (~5.7e-3 vs 1.5e-2 median rel-err) but still approximate -- exact-grid
    consumers use ``vectorized`` / ``cpu-numba``. ``_kde_setup`` raises ``np.linalg.LinAlgError`` on
    a singular covariance exactly as the other backends. See ADR-014 (amended).
    """
    _data, w_n, h11, h12, h22, det, norm = _kde_setup(gk_x_w, gk_y_w, w)
    field = _bin_cic(gk_x_w, gk_y_w, w_n)
    return _fft_convolve_field(field, h11, h12, h22, det, norm)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------


def _chirality_block(model: GhostGkModel) -> dict:
    """Behavioral chirality fingerprint (ADR-037): the model's own extractor + served mean
    on the canonical y-asymmetric probe frame. Emitted into save() metadata; a y-mirrored
    artifact cannot reproduce it (the 4.18.0-weights class of bug). Sparse-frame NaN
    features are fine --- the booster treats NaN as missing and the served (x, y) is
    deterministic, which is all the fingerprint needs."""
    from silly_kicks.tracking._chirality import chirality_fingerprint

    def _predict(frame):
        feats = extract_ghost_gk_features(
            frame, gk_team_id="B", goal_x=105.0, score_diff=0.0, phase=0, ball_carrier_team_id="A"
        )
        return model.predict_mean(feats)

    return chirality_fingerprint(_predict)


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
        self._tree_nodes_y: list[np.ndarray] | None = None
        self._baseline_x: float | None = None
        self._baseline_y: float | None = None
        self._training_gk_x: np.ndarray | None = None
        self._training_gk_y: np.ndarray | None = None
        self._training_leaves: np.ndarray | None = None
        # Recorded training-time sklearn version (preserved across load->save so a
        # parameters-only migration does not launder provenance; spec 2026-07-20 §4).
        self._sklearn_version: str | None = None
        # Optional aggregate corpus provenance (providers / n_games / n_rows), recorded
        # in metadata.json when set; None on a pre-provenance or mechanically-migrated
        # artifact (spec 2026-07-20 §6). Never a per-match id list, never a visibility split.
        self.corpus_provenance: dict | None = None
        # Transient sklearn regressors kept after fit() for the parity gate only
        # (NOT serialized — load() reconstructs from the stored tree node arrays).
        self._sk_reg_x = None
        self._sk_reg_y = None
        # R3 (PR-S81): carrier params used to compute the training team_in_possession,
        # recorded in metadata so serve resolves possession identically. Provenance
        # fields are populated by the trainer before save().
        self.carrier_params: dict = dict(DEFAULT_CARRIER_PARAMS)
        self.training_commit: str | None = None
        self.training_platform: str | None = None

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        *,
        carrier_params: dict | None = None,
    ) -> GhostGkModel:
        """Train on feature matrix + (gk_x, gk_y) labels.

        Parameters
        ----------
        features : pd.DataFrame
            Shape (n_samples, 26).
        labels : pd.DataFrame
            Columns "gk_x", "gk_y" — actual GK positions (goal-relative).
        carrier_params : dict | None
            Ball-carrier scoring params (``tolerance_m``/``beta``/``gamma``) that the
            caller used to compute the training ``team_in_possession`` feature. Recorded
            in metadata (R3) so serve resolves possession identically. ``None`` records
            the library default (:data:`DEFAULT_CARRIER_PARAMS`).

        Examples
        --------
        >>> model = GhostGkModel()
        >>> model.fit(X_train, labels_train)
        """
        self.carrier_params = dict(carrier_params) if carrier_params else dict(DEFAULT_CARRIER_PARAMS)
        from sklearn.ensemble import HistGradientBoostingRegressor

        # Canonical fit-time column order (predict_mean / predict_density index X
        # positionally by feature_idx, so reorder here once at the source).
        X = features[GHOST_GK_FEATURE_NAMES].values.astype(np.float64)
        y_x = labels["gk_x"].values.astype(np.float64)
        y_y = labels["gk_y"].values.astype(np.float64)

        # phase-NUMERIC (spec §3.2): train all features numerically — every split is a
        # num_threshold comparison, so the pickle-free numeric leaf traversal matches
        # sklearn .predict() EXACTLY (no un-serialized categorical routing bitsets), and
        # the latent KDE categorical-routing capability gap is closed by construction.
        def _make_regressor() -> HistGradientBoostingRegressor:
            return HistGradientBoostingRegressor(
                max_iter=self._n_estimators,
                max_depth=self._max_depth,
                categorical_features=None,  # type: ignore[arg-type]  # sklearn stub types this str; None = all-numeric
                random_state=42,
                verbose=self._verbose,
            )

        regressor = _make_regressor()
        regressor.fit(X, y_x)
        regressor_y = _make_regressor()
        regressor_y.fit(X, y_y)

        # Fail-fast on a future sklearn private-API rename (raise, NOT assert — `python -O`
        # strips asserts, which would silently reintroduce the risk this guard prevents).
        # The parity test (test_ghost_gk_serve_mean.py) is the real correctness guard.
        for reg in (regressor, regressor_y):
            if not hasattr(reg, "_predictors") or reg._baseline_prediction.size != 1:
                msg = "sklearn HistGradientBoostingRegressor private API changed — reconstruction needs review"
                raise RuntimeError(msg)

        # Extract tree node arrays for serialization + pickle-free inference (both ensembles).
        self._tree_nodes = [tree_list[0].nodes.copy() for tree_list in regressor._predictors]
        self._tree_nodes_y = [tree_list[0].nodes.copy() for tree_list in regressor_y._predictors]

        # Per-regressor additive baseline (numpy-2 safe: shape (1,1), bare float(ndarray)
        # warns/raises under numpy>=2 — go through .item()).
        self._baseline_x = float(regressor._baseline_prediction.item())
        self._baseline_y = float(regressor_y._baseline_prediction.item())

        # Keep training leaves + labels — the KDE / predict_density still needs them
        # (the gk_x leaf partition + the joint (gk_x, gk_y) labels for the density/mode/spread).
        self._training_leaves = _vectorized_leaf_indices(self._tree_nodes, X)
        self._training_gk_x = np.array(y_x, copy=True)
        self._training_gk_y = np.asarray(y_y, dtype=np.float64).copy()

        # Transient — retained for the parity gate, never serialized (load() reconstructs).
        self._sk_reg_x = regressor
        self._sk_reg_y = regressor_y

        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict the served estimate (x, y) for each sample — the exact boosted HGBR mean.

        Returns the same value the library writes into ``ghost_gk_x/ghost_gk_y``
        (the exact sklearn ``HistGradientBoostingRegressor`` boosted prediction;
        see :meth:`predict_mean`). **Changed in 4.14.0:** previously returned the KDE
        joint *mode*; the mode is now reachable only via
        ``predict_density(...).mode_x/mode_y``. spec (2026-06-04 Option A) §3.4.

        Returns shape (n_samples, 2).

        Examples
        --------
        >>> model.predict(X_test).shape
        (100, 2)
        """
        return self.predict_mean(features)

    def predict_mean(self, features: pd.DataFrame) -> np.ndarray:
        """Served estimate: the exact sklearn HGBR boosted prediction, pickle-free + load-safe.

        ``predict = baseline + sum_trees leaf_value`` (squared-error loss, identity link;
        the learning rate is baked into the stored leaf ``value``s). Reconstructed from
        the serialized tree node arrays + baselines via :func:`_vectorized_leaf_values`,
        so it is deterministic, sklearn-version-independent at inference, and identical
        after :meth:`fit` and :meth:`load`. Cheap — pure leaf traversal, no leaf-match,
        no grid KDE.

        Returns shape (n_samples, 2) — served (x, y).

        Examples
        --------
        >>> model.predict_mean(X_test).shape
        (100, 2)
        """
        if (
            self._tree_nodes is None
            or self._tree_nodes_y is None
            or self._baseline_x is None
            or self._baseline_y is None
        ):
            msg = "Model not fitted. Call .fit() or .load() first."
            raise RuntimeError(msg)

        # Reindex to the canonical fit-time column order (Hyrum guard): the reconstruction
        # indexes X[:, feature_idx] positionally — a reordered DataFrame would silently
        # mis-predict.
        X = features[GHOST_GK_FEATURE_NAMES].values.astype(np.float64)
        out = np.empty((len(X), 2), dtype=np.float64)
        out[:, 0] = self._baseline_x + _vectorized_leaf_values(self._tree_nodes, X)
        out[:, 1] = self._baseline_y + _vectorized_leaf_values(self._tree_nodes_y, X)
        return out

    def predict_density(self, features: pd.DataFrame, *, kde_backend: str = "vectorized") -> list[GhostGkDensity]:
        """Full density prediction per sample.

        Computes leaf co-occurrence weights, weighted 2D KDE, grid evaluation.

        Parameters
        ----------
        kde_backend : {"vectorized", "scipy", "cpu-numba", "fft", "fft-cic"}, default "vectorized"
            KDE kernel. "vectorized" (cpu-numpy) is the default closed-form path; "scipy" is the
            reference oracle; "cpu-numba" runs the serial @njit fused loop (~10x the hot loop,
            value-equivalent within golden tolerance) and requires the ``[numba]`` extra; "fft"
            is the binned-convolution backend (O(k + m log m); ~2000x on the full-k production
            regime) -- faithful on mean/spread and on the mode for UNIMODAL grids, but on near-tie
            MULTIMODAL grids the NGP snap can flip the emitted mode by several metres, and it is NOT
            bit-faithful on the raw ``probabilities`` grid. "fft-cic" adds CIC (bilinear) binning:
            ~76% fewer multimodal mode flips + tighter raw grid than "fft" (NGP) at ~2x the bin
            cost. PREFER "fft-cic" over "fft" for new FFT consumers unless you need NGP's extra speed
            on known-unimodal data; use "vectorized"/"cpu-numba" for an exact raw grid. See ADR-014.

        Examples
        --------
        >>> densities = model.predict_density(X_test)
        >>> densities[0].probabilities.sum()
        1.0
        """
        if self._tree_nodes is None or self._tree_nodes_y is None:
            msg = "Model not fitted. Call .fit() or .load() first."
            raise RuntimeError(msg)
        if self._training_leaves is None or self._training_gk_x is None or self._training_gk_y is None:
            msg = (
                "predict_density is not available on a parameters-only artifact "
                "(distributed artifacts store learned parameters only, not per-sample "
                "training data; spec 2026-07-20). Fit the model locally with .fit() to "
                "use the density path."
            )
            raise RuntimeError(msg)

        training_gk_x = self._training_gk_x
        training_gk_y = self._training_gk_y

        # Reindex to the canonical fit-time column order (same positional guard as
        # predict_mean): the leaf traversal indexes X[:, feature_idx] positionally.
        X = features[GHOST_GK_FEATURE_NAMES].values.astype(np.float64)
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
                    elif kde_backend == "cpu-numba":
                        probs = _kde_density_numba(gk_x_w, gk_y_w, w, grid_points)
                    elif kde_backend == "fft":
                        probs = _kde_density_fft(gk_x_w, gk_y_w, w, grid_points)
                    elif kde_backend == "fft-cic":
                        probs = _kde_density_fft_cic(gk_x_w, gk_y_w, w, grid_points)
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
        - rfcde_weights.npz: gk_x + gk_y tree_nodes_* arrays + baselines +
          training_gk_x/y + training_leaves
        - metadata.json: feature_names, grid_spec, hyperparams, serve_estimator, version
        - SHA256SUMS: per-file integrity hashes

        Option A (4.14.0) format: the npz additionally carries the gk_y tree ensemble
        (``tree_nodes_y_*`` / ``n_trees_y``) and both regressors' additive baselines
        (``baseline_x`` / ``baseline_y``) so :meth:`predict_mean` reconstructs the exact
        boosted prediction after :meth:`load`. ``metadata["version"] == "1.2.0"``.

        Examples
        --------
        >>> model.save(Path("models/ghost_gk_v1"))
        """
        if (
            self._tree_nodes is None
            or self._tree_nodes_y is None
            or self._baseline_x is None
            or self._baseline_y is None
        ):
            msg = "Model not fitted. Call .fit() first."
            raise RuntimeError(msg)

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 1. Weights npz — parameters only (spec 2026-07-20 §2.1). The per-sample training
        # arrays (training_gk_x/y, training_leaves) are STRUCTURALLY never persisted: a
        # distributed artifact carries learned parameters, not the training corpus. The
        # density path (predict_density) needs them and so is available only on a locally
        # fit() model, never a loaded one.
        save_dict: dict[str, np.ndarray] = {
            "n_trees": np.array([len(self._tree_nodes)]),
            # Option A: gk_y ensemble + both additive baselines for the boosted reconstruction.
            "n_trees_y": np.array([len(self._tree_nodes_y)]),
            "baseline_x": np.array([self._baseline_x], dtype=np.float64),
            "baseline_y": np.array([self._baseline_y], dtype=np.float64),
        }
        for i, nodes in enumerate(self._tree_nodes):
            save_dict[f"tree_nodes_{i}"] = nodes.view(np.uint8)
            save_dict[f"tree_dtype_{i}"] = np.array([str(nodes.dtype)], dtype="U2000")
        for i, nodes in enumerate(self._tree_nodes_y):
            save_dict[f"tree_nodes_y_{i}"] = nodes.view(np.uint8)
            save_dict[f"tree_dtype_y_{i}"] = np.array([str(nodes.dtype)], dtype="U2000")

        npz_path = path / "rfcde_weights.npz"
        np.savez_compressed(str(npz_path), **save_dict)  # type: ignore[arg-type]

        # 2. Metadata
        import sklearn

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
            "carrier_params": self.carrier_params,
            # Preserve the recorded training-time version across a load->save migration;
            # only stamp the runtime version for a genuinely fresh fit (spec §4).
            "sklearn_version": self._sklearn_version or sklearn.__version__,
            "training_commit": self.training_commit,
            "training_platform": self.training_platform,
            "corpus_provenance": self.corpus_provenance,
            "serve_estimator": SERVED_ESTIMATOR,
            "version": "1.3.0",
            "stores_training_data": False,
            "chirality": _chirality_block(self),
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
    def load(cls, path: Path, *, legacy_override: bool = False) -> GhostGkModel:
        """Load from local directory with SHA-256 verification.

        No sklearn or onnxruntime needed — tree traversal uses stored
        node arrays with numpy.

        A behavioral chirality fingerprint is enforced (ADR-037 § 9, TF-19 PR-2): a
        pre-PR-2 artifact with no fingerprint is REFUSED unless ``legacy_override=True``
        (which warns), and an output/probe-frame mismatch raises. See ``_chirality``.

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

        # R3 fail-closed: an explicit, conflicting served-estimator tag must not be
        # silently served with this code's read-out. Absent → default (back-compat).
        recorded_estimator = metadata.get("serve_estimator", SERVED_ESTIMATOR)
        if recorded_estimator != SERVED_ESTIMATOR:
            raise IntegrityError(
                f"Model metadata serve_estimator={recorded_estimator!r} != "
                f"code SERVED_ESTIMATOR={SERVED_ESTIMATOR!r}; refusing to serve a "
                f"mismatched estimator (train/serve skew guard)."
            )

        def _load_ensemble(data, nodes_prefix: str, dtype_prefix: str, count_key: str) -> list[np.ndarray]:
            nodes_list = []
            for i in range(int(data[count_key][0])):
                raw_bytes = np.array(data[f"{nodes_prefix}{i}"])
                # ast.literal_eval safely parses the dtype descriptor (list of tuples)
                dtype = np.dtype(ast.literal_eval(str(data[f"{dtype_prefix}{i}"][0])))
                nodes_list.append(raw_bytes.view(dtype))
            return nodes_list

        # Load npz (use context manager to release file handle on Windows)
        with np.load(path / "rfcde_weights.npz", allow_pickle=False) as data:
            # Fail-closed on pre-Option-A artifacts: predict_mean reconstructs the boosted
            # prediction from the gk_y ensemble + baselines, which old (<=4.12) weights lack.
            # Raise a clear "re-fit required" error rather than a cryptic KeyError downstream.
            if "n_trees_y" not in data.files or "baseline_x" not in data.files:
                msg = (
                    "Ghost-GK artifact predates Option A (4.14.0): it lacks the gk_y tree "
                    "ensemble + baselines that predict_mean reconstructs. Re-fit required "
                    "(scripts/train_ghost_gk.py)."
                )
                raise IntegrityError(msg)

            tree_nodes = _load_ensemble(data, "tree_nodes_", "tree_dtype_", "n_trees")
            tree_nodes_y = _load_ensemble(data, "tree_nodes_y_", "tree_dtype_y_", "n_trees_y")
            baseline_x = float(data["baseline_x"][0])
            baseline_y = float(data["baseline_y"][0])

            # Parameters-only artifacts (v1.3.0+) omit the per-sample arrays; a loaded
            # model then has no density path (predict_density raises). Older artifacts
            # still carry them and load unchanged (spec 2026-07-20 §2.2).
            training_gk_x = np.array(data["training_gk_x"]) if "training_gk_x" in data.files else None
            training_gk_y = np.array(data["training_gk_y"]) if "training_gk_y" in data.files else None
            training_leaves = np.array(data["training_leaves"]) if "training_leaves" in data.files else None

        model = cls(
            n_estimators=metadata.get("n_estimators", 500),
            max_depth=metadata.get("max_depth", 8),
        )
        model._tree_nodes = tree_nodes
        model._tree_nodes_y = tree_nodes_y
        model._baseline_x = baseline_x
        model._baseline_y = baseline_y
        model._training_gk_x = training_gk_x
        model._training_gk_y = training_gk_y
        model._training_leaves = training_leaves

        # R3 (PR-S81): consume recorded carrier params; old (v1.0.0) artifacts
        # without the field fall back to the library default.
        model.carrier_params = metadata.get("carrier_params", dict(DEFAULT_CARRIER_PARAMS))
        model.training_commit = metadata.get("training_commit")
        model.training_platform = metadata.get("training_platform")
        model._sklearn_version = metadata.get("sklearn_version")
        model.corpus_provenance = metadata.get("corpus_provenance")

        # Informational provenance only (NOT a correctness guard): inference is
        # sklearn-version-independent — it reads stored npz dtype arrays and imports no
        # sklearn (coupling is fit/extract-time only, spec §3.5). A mismatch flags a
        # re-fit-under-a-new-sklearn for the maintainer's awareness; the parity gate on
        # the fresh fit (Task 12) is the actual guard.
        import sklearn

        recorded_sklearn = metadata.get("sklearn_version")
        if recorded_sklearn is not None and recorded_sklearn != sklearn.__version__:
            warnings.warn(
                f"Ghost-GK artifact fit under sklearn {recorded_sklearn}; runtime is "
                f"{sklearn.__version__}. Inference is sklearn-version-independent (numpy "
                f"reconstruction), so this is informational provenance, not a correctness "
                f"issue. Re-validate parity if re-fitting.",
                stacklevel=2,
            )

        from silly_kicks.tracking._chirality import verify_chirality

        verify_chirality(
            _chirality_block(model),
            metadata.get("chirality"),
            legacy_override=legacy_override,
            model_name="GhostGk",
            error_cls=IntegrityError,  # ghost's own type, so load()'s integrity taxonomy is consistent
        )
        return model

    @classmethod
    def from_variant(cls, variant: GhostGkVariant = "default") -> GhostGkModel:
        """Load a model variant by name.

        ``"default"`` is bundled in the wheel. ``"full"`` is downloaded
        from HuggingFace Hub on first use (requires ``[ghost-gk]`` extra).

        Parameters
        ----------
        variant : "default" | "full"
            ``"default"``: lightweight model (~12 MB, 36 k training samples).
            ``"full"``: high-resolution model (~170 MB, 887 k training samples).

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
    carrier: pd.DataFrame | None = None,
    link_frame_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Per-frame ghost-GK primitive (batched).

    Adds ghost_gk_x, ghost_gk_y columns.
    One prediction per (frame, GK team). Results written to GK rows.

    ``ghost_gk_x/y`` are the served boosted-mean position (the exact sklearn HGBR
    prediction via :meth:`GhostGkModel.predict_mean`), in goal-relative coordinates
    (x = distance from the defended goal line), clamped to the physical pitch
    (x in [0, 105], y in [0, 68]; 4.22.1 -- garbage input, e.g. a mis-flagged
    ``is_goalkeeper``, can push the regressor outside its trained domain; a clamp
    emits a warning and only ever fires on such rows). The served position is a pure
    leaf-value tree traversal (no KDE, no per-sample data), so it works on any loaded
    parameters-only artifact. The density-dispersion read-out (``predict_density``)
    was retired from this per-frame surface with the per-sample arrays
    (spec 2026-07-20 §3.1); it survives only on a locally ``fit()`` model.

    Input frames MUST be in LTR-normalized convention (home team attacks
    right in all periods --- standard silly-kicks tracking output).

    Parameters
    ----------
    frames : pd.DataFrame
        Tracking frames (TRACKING_FRAMES_COLUMNS schema, LTR-normalized).
    model : GhostGkModel | "default" | "full" | None
        ``"default"`` / ``None``: bundled lightweight model (~12 MB).
        ``"full"``: high-resolution model (~170 MB, downloaded from HF Hub).
        Or a pre-loaded ``GhostGkModel`` instance.
    home_team_id : int | str
        Home team ID (attacks right -> defends at x=0).
    actions : pd.DataFrame | None
        SPADL actions for score_diff and phase context. If None, both
        default to 0 (backward-compatible with 3.19.0 behaviour).
    carrier : pd.DataFrame | None, default None
        Optional precomputed carrier --- the
        ``["game_id","period_id","frame_id","ball_carrier_team_id"]`` projection of
        ``infer_ball_carrier(frames, **model.carrier_params)`` on the FULL frames.
        When None, computed internally. Supply it to avoid recomputation across
        repeated ghost-GK calls or across families that also resolve possession
        (mirrors ``links``). Must be computed on full frames with the model's
        carrier_params for the byte-identical frame-restriction invariant to hold.
    link_frame_ids : set[int] | None, default None
        When provided, restrict the served predictions to GK samples whose
        ``frame_id`` is in this set. Feature extraction still runs over the FULL
        frames, so the per-period defending-goal mean-x and the cross-period
        one-step velocity state are preserved exactly --- predictions are per-sample
        independent, so the restricted result is byte-identical to the unrestricted
        one for the kept frames. When None, every sample is predicted
        (backward-compatible). See PR-S66 spec sections 2-3.

    Returns
    -------
    pd.DataFrame
        Copy of frames with ghost_gk_x, ghost_gk_y added.

    Examples
    --------
    >>> from silly_kicks.tracking._ghost_gk import compute_ghost_gk
    >>> result = compute_ghost_gk(frames, home_team_id=1)
    """
    out = frames.copy()
    out["ghost_gk_x"] = np.nan
    out["ghost_gk_y"] = np.nan

    _resolved, meta, _batch_features, positions, _clamped = _serve_positions_core(
        frames,
        model=model,
        home_team_id=home_team_id,
        actions=actions,
        carrier=carrier,
        link_frame_ids=link_frame_ids,
    )

    if len(positions) == 0:
        return out

    # Build result DataFrame from predictions (single merge, not O(n*m) loop)
    result_df = pd.DataFrame(
        {
            "game_id": meta["game_id"].values,
            "period_id": meta["period_id"].values,
            "frame_id": meta["frame_id"].values,
            "team_id": meta["gk_team_id"].values,
            "ghost_gk_x": positions[:, 0],
            "ghost_gk_y": positions[:, 1],
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

    return out


def _serve_positions_core(
    frames: pd.DataFrame,
    *,
    model: GhostGkModel | GhostGkVariant | None,
    home_team_id: int | str,
    actions: pd.DataFrame | None,
    carrier: pd.DataFrame | None,
    link_frame_ids: set[int] | None,
) -> tuple[GhostGkModel, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Shared serve: ``(resolved, meta, batch_features, positions, clamped_mask)``.

    Single-sources model resolution, context callbacks, feature extraction, the 4.12.1
    duplicate-(frame, gk_team) collapse, ``predict_mean`` and the 4.22.1 physical-pitch
    clamp. Returns the goal-relative positions AND the per-row clamp mask captured
    BEFORE ``np.clip`` --- the information ``compute_ghost_gk`` discards.

    The resolved model rides along so the caller can serve positions without
    re-resolving. ``batch_features`` is also returned for callers that need the
    extracted feature set (historically the retired ``predict_density`` pass); the
    positions themselves come from ``predict_mean(batch_features)`` inside this core.
    """
    resolved = _resolve_model(model)

    # Build context callbacks from actions
    score_fn = _build_score_lookup(actions, home_team_id) if actions is not None else None
    phase_fn = _build_phase_lookup(actions) if actions is not None else None

    # PR-S81 serve-carrier consistency: compute the carrier on FULL frames with the
    # model's recorded carrier_params (R3) so team_in_possession matches training. The
    # carrier lookup in _extract_one_frame is per-(game,period,frame) independent, so
    # restricting extraction to link_frame_ids stays byte-identical for kept frames
    # (mirrors xS _xshot_occurrence.py). A caller may supply a precomputed `carrier`
    # (computed on full frames with this model's carrier_params) to skip the internal
    # inference (N5 cache convention; mirrors `links`).
    if carrier is None:
        carrier_raw = infer_ball_carrier(frames, **resolved.carrier_params)
        carrier = carrier_raw[["game_id", "period_id", "frame_id", "ball_carrier_team_id"]]

    # PR-S66 §5: restrict the heavy feature extraction to the action-linked frames.
    # _extract_all_ghost_gk_features still walks every frame to maintain the
    # cross-period one-step velocity state and computes the per-period defending-goal
    # mean-x over the full frames, so the linked-frame features --- and the per-sample
    # predict_mean, which has zero cross-sample coupling --- are byte-identical to the
    # unrestricted compute. See TestExtractionRestriction.
    batch_features, meta = _extract_all_ghost_gk_features(
        frames,
        home_team_id=home_team_id,
        carrier=carrier,
        score_at_time=score_fn,
        phase_at_time=phase_fn,
        link_frame_ids=link_frame_ids,
    )

    if len(batch_features) == 0:
        return resolved, meta, batch_features, np.empty((0, 2), dtype=float), np.empty(0, dtype=bool)

    # Collapse duplicate (frame, gk_team) inference samples (4.12.1 fix). Two same-team
    # is_goalkeeper rows in one frame (a rostered backup keeper carried on-pitch
    # alongside the starter, or a GK-substitution overlap frame) make
    # _extract_all_ghost_gk_features emit one sample per GK row, all keyed on
    # (game, period, frame, gk_team_id). The features are byte-identical per
    # (frame, gk_team) — only the per-GK-row label differs, and labels are unused
    # here — so collapsing to one sample keeps the prediction single-pass AND keeps the
    # downstream left-merge 1:1 with the GK rows (duplicate result_df keys would
    # inflate the merge and length-mismatch the positional assignment below). The
    # training builder keeps per-GK-row samples (distinct labels) untouched.
    _key_cols = ["game_id", "period_id", "frame_id", "gk_team_id"]
    _keep = ~meta.duplicated(subset=_key_cols, keep="first")
    if not _keep.all():
        keep_mask = _keep.to_numpy()
        meta = meta[keep_mask].reset_index(drop=True)
        batch_features = batch_features[keep_mask].reset_index(drop=True)

    # Batch predict: position = served boosted mean (cheap leaf-value traversal).
    positions = resolved.predict_mean(batch_features)

    # 4.22.1 (lakehouse report 2026-06-11 item 2): clamp the served position to the
    # PHYSICAL pitch in goal-relative coords (x = distance from the defended goal
    # line). Garbage input (e.g. a mis-flagged is_goalkeeper upstream) can wrong-foot
    # the per-period goal-side flip and push the boosted regressor far outside its
    # trained label domain -- a keeper served behind the goal line is never physically
    # meaningful. Physical bounds, NOT the trained grid domain: healthy extrapolation
    # slightly past GRID_X_MAX (a sweeper rush) must stay byte-unchanged. The clamp
    # lives at this serving seam so GhostGkModel.predict_mean keeps its exact-boosted
    # parity contract (ADR-016).
    _lo = np.array([0.0, 0.0])
    _hi = np.array([_FIELD_LENGTH, _FIELD_WIDTH])
    # Captured BEFORE np.clip -- after clipping the per-row information is
    # unrecoverable, and the probe contract requires a non-null per-row boolean.
    clamped = np.asarray(((positions < _lo) | (positions > _hi)).any(axis=1), dtype=bool)
    if bool(clamped.any()):
        # P7: stacklevel is 3, NOT the original 2. The warning has moved one frame deeper
        # (user -> compute_ghost_gk -> _serve_positions_core -> warn), so stacklevel=2
        # would now point at library internals instead of the caller. 3 is correct for
        # BOTH public entry points, since serve_ghost_gk_positions sits at the same depth.
        # P8: category added while re-homing. The message is UNCHANGED (Chesterton), but
        # the warning is now emitted from a SECOND public entry point, and a consumer
        # wanting to silence the batch-clamp notice should not have to silence every
        # UserWarning from tracking.
        warnings.warn(
            "ghost-GK: one or more served positions fell outside the physical pitch and "
            "were clamped; suspect upstream tracking quality (e.g. a mis-flagged "
            "is_goalkeeper).",
            GhostClampWarning,
            stacklevel=3,
        )
        positions = np.clip(positions, _lo, _hi)
    return resolved, meta, batch_features, positions, clamped


def serve_ghost_gk_positions(
    frames: pd.DataFrame,
    *,
    model: GhostGkModel | GhostGkVariant | None = None,
    home_team_id: int | str,
    actions: pd.DataFrame | None = None,
    carrier: pd.DataFrame | None = None,
    link_frame_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Serve ghost-GK positions ONLY, with per-row clamp / out-of-training-box provenance.

    Positions-only sibling of :func:`compute_ghost_gk`: it skips the KDE density pass
    (the entire cost driver) and, unlike ``compute_ghost_gk``, returns the per-row
    ``ghost_clamped`` mask instead of collapsing it into one batch warning. Coordinates
    are GOAL-RELATIVE (``x`` = distance from the defended goal line) --- the caller does
    the write-back to frame coordinates.

    ``ghost_out_of_box`` marks positions beyond the ghost's trained label hull
    (``GRID_X_MAX`` = 30 m) and is evaluated on goal-relative ``x`` BEFORE any flip.

    The goal-relative contract above is prose; the gate that enforces it is
    ``tests/gkdv/test_engine.py::test_out_of_box_flag_keys_on_GOAL_RELATIVE_x_and_survives_writeback``.

    Parameters
    ----------
    frames : pd.DataFrame
        LTR-normalized tracking frames (see :func:`compute_ghost_gk`).
    model, home_team_id, actions, carrier, link_frame_ids
        As :func:`compute_ghost_gk`.

    Returns
    -------
    pd.DataFrame
        One row per ``(game_id, period_id, frame_id, gk_team_id)`` with ``ghost_gr_x``,
        ``ghost_gr_y``, ``ghost_clamped``, ``ghost_out_of_box``.

    Examples
    --------
    >>> out = serve_ghost_gk_positions(frames, home_team_id=1)  # doctest: +SKIP
    >>> bool(out["ghost_clamped"].notna().all())  # doctest: +SKIP
    True
    """
    _resolved, meta, _batch_features, positions, clamped = _serve_positions_core(
        frames,
        model=model,
        home_team_id=home_team_id,
        actions=actions,
        carrier=carrier,
        link_frame_ids=link_frame_ids,
    )
    if len(positions) == 0:
        # The empty frame's join-key dtypes MUST match the populated path's, or a
        # pd.concat across a per-match loop where one match has no detected GK silently
        # degrades period_id/frame_id from int64 to object -- and the caller joins on
        # exactly these columns (ADR-019 class defect).
        #
        # The four join keys are therefore DERIVED FROM THE INPUT, not hard-coded: they
        # are not fixed by any schema. `game_id`/`team_id` are int64 for a native provider
        # and object for the kloppy-family ones, so a hard-coded pair is only ever right
        # for the provider it was written against. It was measured wrong on both:
        # `game_id` was pinned object while the populated path yields pandas 3's `str`,
        # and `gk_team_id` was pinned float64 for every provider whose team ids are not.
        #
        # The derivation goes through ONE REAL VALUE rather than a zero-row column slice,
        # because the populated path builds from python scalars (`pd.DataFrame(meta_rows)`)
        # and that INFERENCE is not always the source column's own dtype: pandas 3 infers
        # `str` from an object-dtype string column. Round-tripping a single non-null value
        # through the same construction reproduces the populated dtype by definition; a
        # slice would silently re-introduce the very mismatch this branch exists to avoid.
        def _empty_join_key(column: str, fallback: str) -> pd.Series:
            if column not in frames.columns:
                return pd.Series(dtype=fallback)
            observed = frames[column].dropna()
            if observed.empty:
                # Nothing to infer from -- the column slice is the closest available truth.
                return frames[column].iloc[:0].reset_index(drop=True)
            return pd.DataFrame([{column: observed.iloc[0]}])[column].iloc[:0].reset_index(drop=True)

        return pd.DataFrame(
            {
                "game_id": _empty_join_key("game_id", "object"),
                "period_id": _empty_join_key("period_id", "int64"),
                "frame_id": _empty_join_key("frame_id", "int64"),
                # The GK's team id comes off the frames' `team_id` column in the
                # populated path (`meta["gk_team_id"]` is filled from `gk_row["team_id"]`).
                "gk_team_id": _empty_join_key("team_id", "float64"),
                "ghost_gr_x": pd.Series(dtype=float),
                "ghost_gr_y": pd.Series(dtype=float),
                "ghost_clamped": pd.Series(dtype=bool),
                "ghost_out_of_box": pd.Series(dtype=bool),
            }
        )
    return pd.DataFrame(
        {
            "game_id": meta["game_id"].to_numpy(),
            "period_id": meta["period_id"].to_numpy(),
            "frame_id": meta["frame_id"].to_numpy(),
            "gk_team_id": meta["gk_team_id"].to_numpy(),
            "ghost_gr_x": positions[:, 0],
            "ghost_gr_y": positions[:, 1],
            "ghost_clamped": clamped.astype(bool),
            "ghost_out_of_box": (positions[:, 0] > GRID_X_MAX),
        }
    )
