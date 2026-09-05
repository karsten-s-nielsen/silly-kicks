"""TF-60 PR5 --- ghost-outfield model (rest-defense rearguard positioning).

A league-average, POSSESSION-CONDITIONED positioning model for a team's deepest field defenders. It
mirrors :class:`~silly_kicks.tracking._ghost_gk.GhostGkModel` (HistGradientBoostingRegressor
boosted-mean x/y ensembles, served as the exact pickle-free reconstruction; parameters-only npz + JSON
+ SHA256 serialization; chirality + feature-contract fail-closed load-guards; numba leaf walk), but
predicts an **individual** rearguard player keyed by a deterministic lateral-rank ``slot`` within the
deepest-``n`` defenders, instead of the single goalkeeper.

Like ghost-GK it conditions on possession: ONE model serves both regimes, discriminated by the live
``team_in_possession`` feature. Training (``both_teams=True``) fits BOTH teams' deepest-n per frame ---
the ball-carrier's rest-defense **rearguard** (the deep cover it keeps while attacking, to blunt the
counter; ``team_in_possession=1``) AND the other team's **defensive line** facing the attack
(``team_in_possession=0``). The rest-defense application (this PR's driver, and PR6's arm) serves the
in-possession slice; other analyses can query the out-of-possession regime.

It exists to feed PR6's outfield counterfactual arm: substitute a team's actual rearguard with these
league-average "ghost" positions and difference the pitch-control / accessible-space fields.

**The leakage rule (spec 5).** The prediction target is slot-``K``'s own ``(x, y)``, so **no feature
may encode team A's rearguard coordinates or geometry** (the target set) --- a positional rearguard
summary (line-x, deepest-defender, compactness/width) contains slot-``K`` and would leak the label.
The model therefore conditions ONLY on the ball, the opponent's counter-threat geometry, game context,
and the ``slot_index`` (a lateral RANK, not a coordinate summary --- exempt). Situational velocity only
(ball + opponent mass); never the slot player's own velocity (a "league-average given the situation"
ghost must not be told which way *this* player is moving).

Unlike ghost-GK there is **no grid, no KDE density, and no label-domain cap** --- the mean serve is the
only read-out, and a rearguard player's honest position anywhere is a valid label.

See NOTICE for full bibliographic citations (Le et al. 2017 ghosting; DEFCON-GNN comparator).
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from silly_kicks._polygon import as_polygon
from silly_kicks.id_compat import canonical_id, ids_match, same_id
from silly_kicks.spadl import config as spadlconfig

from . import _geometry as _geo
from ._chirality import verify_chirality
from ._defensive_line import select_back_line_players
from ._feature_contract import verify_feature_contract
from ._ghost_gk import (
    _build_phase_lookup,
    _build_score_lookup,
    _flatten_trees,
    _vectorized_leaf_values,
)
from ._gk_resolve import GoalMap, resolve_defended_goals
from ._velocity_availability import (
    variant_key_for_velocity as _variant_key_for_velocity,
)
from ._velocity_availability import (
    velocity_availability_is_mixed as _velocity_availability_is_mixed,
)
from ._velocity_availability import (
    velocity_unavailable_by_design as _velocity_unavailable_by_design,
)
from ._visibility import region_observed_fraction

_FIELD_LENGTH: float = float(spadlconfig.field_length)
_FIELD_WIDTH: float = float(spadlconfig.field_width)
_GOAL_Y: float = _FIELD_WIDTH / 2.0
# Opponent counter-threat spearhead: the k B-outfield players nearest A's defended goal define the
# "forward" centroid + its closing velocity (frozen extractor parameter, spec 5 / 17).
_N_OPP_FORWARD: int = 4
# A's defensive third boundary in goal-relative x (0 = A's defended goal line).
_DEF_THIRD_GR_X: float = _FIELD_LENGTH / 3.0

# ---------------------------------------------------------------------------
# Feature vector (frozen, leakage-safe --- spec 5)
# ---------------------------------------------------------------------------
# 20 faithful features. All geometry is goal-relative to A's DEFENDED goal (the counter-attack
# target), oriented by GoalMap/ADR-055. NONE is derived from A's rearguard positions (the leakage
# rule); the only rearguard-derived feature is ``slot_index``, a lateral RANK (exempt).
GHOST_OUTFIELD_FEATURE_NAMES: list[str] = [
    # Ball state (A's ball)
    "ball_x",
    "ball_y",
    "ball_vx",
    "ball_vy",
    "ball_speed",
    "ball_distance_to_own_goal",
    "ball_to_own_goal_angle",
    "ball_in_own_half",
    # Opponent (B) counter-threat geometry --- leakage-safe (B is not the target set)
    "opp_in_def_third_count",
    "opp_deepest_x",
    "opp_forward_centroid_x",
    "opp_forward_centroid_y",
    "ball_to_deepest_opp_dist",
    "opp_forward_centroid_vx",
    # Game context
    "phase",
    "team_in_possession",
    "score_diff",
    "time_seconds",
    "period_id",
    # Slot (lateral rank 1..n) --- the multi-agent feature
    "slot_index",
]

GhostOutfieldFeatureSet = Literal["faithful", "position_only"]

# Position-only variant: drop the situational velocity features so a fitted model scores on a lone
# velocity-less SB360 freeze frame. Dropped (shorter vector), never NaN-filled (feature contract).
# NB: the slot player's OWN velocity is deliberately NOT a feature (see the module leakage rule), so
# the only velocity features are situational (ball + opponent mass).
_GHOST_OUTFIELD_VELOCITY_FEATURES: tuple[str, ...] = (
    "ball_vx",
    "ball_vy",
    "ball_speed",
    "opp_forward_centroid_vx",
)

GHOST_OUTFIELD_FEATURE_NAMES_POSITION_ONLY: list[str] = [
    f for f in GHOST_OUTFIELD_FEATURE_NAMES if f not in _GHOST_OUTFIELD_VELOCITY_FEATURES
]  # 16


# ---------------------------------------------------------------------------
# Feature extraction (leakage-safe; per-frame, per-slot)
# ---------------------------------------------------------------------------

_OUTFIELD_KEY_COLS: list[str] = ["game_id", "frame_id", "team_id", "player_id"]
_OUTFIELD_TARGET_COLS: list[str] = ["target_x", "target_y"]


def _feature_names_for(feature_set: GhostOutfieldFeatureSet) -> list[str]:
    """The model's feature columns for its feature set (20 faithful / 16 position_only)."""
    return (
        GHOST_OUTFIELD_FEATURE_NAMES_POSITION_ONLY if feature_set == "position_only" else GHOST_OUTFIELD_FEATURE_NAMES
    )


def _extract_output_columns(feature_set: GhostOutfieldFeatureSet) -> list[str]:
    """Column order of the extractor output (features + keys + target).

    ``period_id`` and ``slot_index`` live inside the feature names and double as join keys, so they
    are NOT repeated in the key list.
    """
    return list(_feature_names_for(feature_set)) + _OUTFIELD_KEY_COLS + _OUTFIELD_TARGET_COLS


def _gr_x(x: float, flip: bool) -> float:
    """Goal-relative x under the ADR-051 180-degree point reflection (flip iff the goal is at high x)."""
    return (_FIELD_LENGTH - x) if flip else x


def _gr_y(y: float, flip: bool) -> float:
    """Goal-relative y under the point reflection (BOTH axes flip; ADR-051 8b)."""
    return (_FIELD_WIDTH - y) if flip else y


def _gr_vel(v: float, flip: bool) -> float:
    """Point reflection NEGATES a velocity component (vx and vy alike)."""
    return -v if flip else v


def _rows_for_team(
    frame: pd.DataFrame,
    gid: Any,
    pid: Any,
    fid: Any,
    t_team: Any,
    carrier_team: Any,
    *,
    home_team_id: int | str,
    gmap: GoalMap,
    phase_at: Callable[[Any, float], int] | None,
    score_at: Callable[[Any, float], float] | None,
    has_vel: bool,
    n_rearguard: int,
    names: list[str],
) -> list[dict]:
    """Per-(frame, modeled team ``t_team``) leakage-safe feature rows (one per rearguard slot).

    ``t_team``'s deepest-``n`` rearguard, goal-relative to ``t_team``'s DEFENDED goal; the opponent is
    the OTHER team's outfield (``t_team``'s counter-threat when it is in possession, the attackers
    bearing down when it is not). ``team_in_possession`` = 1 iff ``t_team`` is the ball-carrier's team,
    else 0 --- the live possession discriminator that makes ONE model serve both regimes (ADR-055 /
    spec 5). Empty list on an unresolvable defended goal or an empty rearguard (ADR-027).
    """
    goal_x = gmap.get(gid, pid, t_team, allow_guess=True)
    if goal_x is None:
        return []
    flip = goal_x > 50.0
    non_ball = frame[~frame["is_ball"].astype(bool)]

    # --- ball state (goal-relative to t_team's defended goal) ---
    ball = frame[frame["is_ball"].astype(bool)]
    if len(ball) > 0:
        bx = float(ball["x"].iloc[0])
        by = float(ball["y"].iloc[0])
        bvx = float(ball["vx"].iloc[0]) if has_vel else np.nan
        bvy = float(ball["vy"].iloc[0]) if has_vel else np.nan
    else:
        bx = by = bvx = bvy = np.nan
    ball_x = _gr_x(bx, flip)
    ball_y = _gr_y(by, flip)
    ball_vx = _gr_vel(bvx, flip)
    ball_vy = _gr_vel(bvy, flip)
    ball_speed = float(np.hypot(bvx, bvy)) if (has_vel and not np.isnan(bvx)) else np.nan
    ball_dist = float(np.hypot(ball_x, ball_y - _GOAL_Y)) if not np.isnan(ball_x) else np.nan
    ball_angle = float(np.arctan2(ball_y - _GOAL_Y, ball_x)) if not np.isnan(ball_x) else np.nan
    ball_own_half = 1.0 if (not np.isnan(ball_x) and ball_x < _FIELD_LENGTH / 2.0) else 0.0

    # --- opponent (the OTHER team) counter-threat geometry (leakage-safe: not t_team's rearguard) ---
    t_mask = ids_match(non_ball["team_id"], t_team)
    gk_mask = non_ball["is_goalkeeper"].astype(bool)
    opp = non_ball[~t_mask & ~gk_mask]
    if len(opp) > 0:
        ox = opp["x"].to_numpy(dtype="float64")
        oy = opp["y"].to_numpy(dtype="float64")
        opp_grx = (_FIELD_LENGTH - ox) if flip else ox
        opp_gry = (_FIELD_WIDTH - oy) if flip else oy
        opp_in_third = int(np.sum(opp_grx < _DEF_THIRD_GR_X))
        order = np.argsort(opp_grx)  # nearest t_team's defended goal first
        fwd = order[: min(_N_OPP_FORWARD, len(opp))]
        opp_deepest_x = float(opp_grx[order[0]])
        opp_deepest_y = float(opp_gry[order[0]])
        opp_cx = float(np.mean(opp_grx[fwd]))
        opp_cy = float(np.mean(opp_gry[fwd]))
        if has_vel:
            ovx = opp["vx"].to_numpy(dtype="float64")
            opp_gvx = (-ovx) if flip else ovx
            opp_fwd_vx = float(np.mean(opp_gvx[fwd]))
        else:
            opp_fwd_vx = np.nan
        ball_to_deepest = (
            float(np.hypot(ball_x - opp_deepest_x, ball_y - opp_deepest_y)) if not np.isnan(ball_x) else np.nan
        )
    else:
        opp_in_third = 0
        opp_deepest_x = opp_cx = opp_cy = opp_fwd_vx = ball_to_deepest = np.nan

    # --- context ---
    time_s = float(frame["time_seconds"].iloc[0]) if "time_seconds" in frame.columns else 0.0
    phase_v = float(phase_at(gid, time_s)) if phase_at is not None else 0.0
    if score_at is not None:
        sd = score_at(gid, time_s)
        if not same_id(t_team, home_team_id):
            sd = -sd
    else:
        sd = 0.0
    team_in_poss = 1.0 if same_id(t_team, carrier_team) else 0.0

    base = {
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_vx": ball_vx,
        "ball_vy": ball_vy,
        "ball_speed": ball_speed,
        "ball_distance_to_own_goal": ball_dist,
        "ball_to_own_goal_angle": ball_angle,
        "ball_in_own_half": ball_own_half,
        "opp_in_def_third_count": float(opp_in_third),
        "opp_deepest_x": opp_deepest_x,
        "opp_forward_centroid_x": opp_cx,
        "opp_forward_centroid_y": opp_cy,
        "ball_to_deepest_opp_dist": ball_to_deepest,
        "opp_forward_centroid_vx": opp_fwd_vx,
        "phase": phase_v,
        "team_in_possession": team_in_poss,
        "score_diff": float(sd),
        "time_seconds": time_s,
        # Read the period from the frame (a number), NOT the groupby key `pid` (typed Hashable); this
        # base value is overwritten by the raw `pid` join key in the per-slot loop below.
        "period_id": float(frame["period_id"].iloc[0]),
    }

    # --- rearguard slots (lateral rank in goal-relative frame) ---
    rearguard = select_back_line_players(frame, t_team, goal_x < 50.0, n=n_rearguard)
    if len(rearguard) == 0:
        return []
    rg = rearguard.copy()
    rg["_gry"] = (_FIELD_WIDTH - rg["y"]) if flip else rg["y"]
    rg = rg.sort_values("_gry", kind="stable").reset_index(drop=True)

    out_rows: list[dict] = []
    for slot0, (_, prow) in enumerate(rg.iterrows()):
        rowd: dict[str, object] = {c: base[c] for c in names if c != "slot_index"}
        rowd["slot_index"] = float(slot0 + 1)
        rowd["game_id"] = gid
        rowd["period_id"] = pid
        rowd["frame_id"] = fid
        rowd["team_id"] = t_team
        rowd["player_id"] = prow["player_id"]
        rowd["target_x"] = _gr_x(float(prow["x"]), flip)
        rowd["target_y"] = _gr_y(float(prow["y"]), flip)
        out_rows.append(rowd)
    return out_rows


def _extract_all_ghost_outfield_features(
    frames: pd.DataFrame,
    actions: pd.DataFrame | None,
    *,
    home_team_id: int | str,
    carrier: pd.DataFrame | None = None,
    feature_set: GhostOutfieldFeatureSet = "faithful",
    goal_map: GoalMap | None = None,
    n_rearguard: int = 4,
    both_teams: bool = False,
) -> pd.DataFrame:
    """Batch-extract per-(frame, team, rearguard-slot) leakage-safe features + goal-relative targets.

    Possession-CONDITIONED (ADR-055): one model serves both regimes, discriminated by the live
    ``team_in_possession`` feature. ``both_teams=True`` (TRAINING) emits BOTH non-ball teams' deepest-n
    per frame --- the ball-carrier's rest-defense rearguard (``team_in_possession=1``) AND the other
    team's defensive line facing the attack (``team_in_possession=0``); ``both_teams=False`` (SERVING /
    the default: rest defense wants only the carrier's rearguard) emits just the ball-carrier team.

    Features are ALWAYS goal-relative to the MODELED team's DEFENDED goal, via the ADR-051 180-degree
    point reflection, and the opponent geometry is the OTHER team's outfield --- so a single leakage-safe
    vector (ball + opponent counter-threat + context + slot index; NEVER a quantity derived from the
    modeled team's own rearguard, the target set) describes both regimes. Honest-NaN (no rows) when the
    ball-carrier, the defended goal, or a two-team frame is unresolvable (never a fabricated row; ADR-027).

    Returns one row per ``(game_id, period_id, frame_id, team_id, slot_index)`` carrying the feature
    columns, the bookkeeping key ``player_id`` (the actual player in the slot --- NOT a model feature),
    and the goal-relative label ``(target_x, target_y)``.
    """
    out_cols = _extract_output_columns(feature_set)
    names = _feature_names_for(feature_set)
    if len(frames) == 0:
        return pd.DataFrame(columns=out_cols)

    gmap: GoalMap = goal_map if goal_map is not None else resolve_defended_goals(frames)
    score_at: Callable[[Any, float], float] | None = (
        _build_score_lookup(actions, home_team_id) if actions is not None and len(actions) else None
    )
    phase_at: Callable[[Any, float], int] | None = (
        _build_phase_lookup(actions) if actions is not None and len(actions) else None
    )
    has_vel = "vx" in frames.columns and "vy" in frames.columns

    carrier_idx: pd.Series | None = None
    if carrier is not None and "ball_carrier_team_id" in carrier.columns:
        carrier_idx = carrier.set_index(["game_id", "period_id", "frame_id"])["ball_carrier_team_id"]

    rows: list[dict] = []
    for (gid, pid, fid), frame in frames.groupby(["game_id", "period_id", "frame_id"], sort=True):
        # --- resolve the ball-carrier (in-possession) team ---
        carrier_team: Any = None
        if carrier_idx is not None:
            try:
                carrier_team = carrier_idx.get((gid, pid, fid))
            except (KeyError, TypeError):
                carrier_team = None
        if (carrier_team is None or pd.isna(carrier_team)) and "team_in_possession" in frame.columns:
            _vals = frame["team_in_possession"].dropna()
            carrier_team = _vals.iloc[0] if len(_vals) else None
        if carrier_team is None or pd.isna(carrier_team):
            continue

        non_ball = frame[~frame["is_ball"].astype(bool)]
        real_teams = list(non_ball["team_id"].dropna().unique())
        if len({canonical_id(t) for t in real_teams}) != 2:
            continue

        modeled = real_teams if both_teams else [carrier_team]
        for t_team in modeled:
            rows.extend(
                _rows_for_team(
                    frame,
                    gid,
                    pid,
                    fid,
                    t_team,
                    carrier_team,
                    home_team_id=home_team_id,
                    gmap=gmap,
                    phase_at=phase_at,
                    score_at=score_at,
                    has_vel=has_vel,
                    n_rearguard=n_rearguard,
                    names=names,
                )
            )

    if not rows:
        return pd.DataFrame(columns=out_cols)
    return pd.DataFrame(rows)[out_cols]


# ---------------------------------------------------------------------------
# Model artifact + serving
# ---------------------------------------------------------------------------

SERVED_ESTIMATOR = "boosted_mean"

#: Valid model variant names.
GhostOutfieldVariant = Literal["default", "position_only"]

_ENV_VAR = "SILLY_KICKS_GHOST_OUTFIELD_PATH"
_WEIGHTS_ROOT = Path(__file__).parent / "_ghost_outfield_weights"
_HF_REPO_ID = "silly-kicks/ghost-outfield-v1"


class IntegrityError(Exception):
    """Raised when a ghost-outfield artifact fails SHA-256 / chirality / feature-contract checks."""


def _outfield_probe_frame() -> pd.DataFrame:
    """One synthetic 2-team frame, deliberately y-ASYMMETRIC, team A in possession, defending x=0.

    Both keepers are present (GoalMap resolution) + ``team_in_possession`` is set + A's deepest-4 are
    well separated, so the extractor yields 4 finite per-slot rows. A y-mirrored artifact cannot
    reproduce the served outputs on it (the chirality property). Distinct from the ghost-GK canonical
    frame, which carries no A keeper and no possession column and so cannot drive this GoalMap-based
    extractor.
    """

    def r(pid, team, x, y, *, gk=False, ball=False, vx=0.3, vy=-0.2):
        return {
            "game_id": "ofprobe",
            "period_id": 1,
            "frame_id": 1,
            "time_seconds": 10.0,
            "team_id": (None if ball else team),
            "player_id": (None if ball else pid),
            "is_ball": ball,
            "is_goalkeeper": gk,
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "team_in_possession": "A",
        }

    rows = [
        r("AGK", "A", 3.0, 30.0, gk=True),
        r("A1", "A", 18.0, 12.0),
        r("A2", "A", 20.0, 28.0),
        r("A3", "A", 22.0, 45.0),
        r("A4", "A", 25.0, 58.0),
        r("A5", "A", 45.0, 20.0),
        r("BGK", "B", 102.0, 38.0, gk=True),
        r("B1", "B", 20.0, 25.0),
        r("B2", "B", 30.0, 40.0),
        r("B3", "B", 55.0, 30.0),
        r("B4", "B", 70.0, 50.0),
        r("B5", "B", 88.0, 22.0),
        r(None, None, 55.0, 30.0, ball=True, vx=1.5, vy=0.5),
    ]
    df = pd.DataFrame(rows)
    df["is_ball"] = df["is_ball"].astype(bool)
    return df


def _outfield_probe_actions() -> pd.DataFrame:
    """Minimal actions for the probe: one pass, no shots/set-pieces -> score 0, phase 0."""
    return pd.DataFrame(
        [
            {
                "game_id": "ofprobe",
                "period_id": 1,
                "time_seconds": 5.0,
                "team_id": "A",
                "player_id": "A5",
                "type_name": "pass",
                "result_name": "success",
                "start_x": 45.0,
                "start_y": 20.0,
            }
        ]
    )


def _canonical_frame_sha(frame: pd.DataFrame) -> str:
    """SHA-256 of the probe frame, pandas-major-INVARIANT (ADR-057).

    ``frame.to_dict("records")`` yields native Python scalars on pandas 2 but numpy scalars on
    pandas 3, and ``json.dumps(..., default=str)`` then serializes a native float as the JSON number
    ``10.0`` but a numpy float (routed through ``default=str``) as the JSON STRING ``"10.0"`` -- a
    different digest per pandas major. A bundled artifact whose chirality frame-hash was recorded
    under one major would then fail its load-time frame-hash check under the other (measured: the
    bundled ghost-outfield weights load under pandas 2.3.3 and fail under pandas 3.0.2), and CI spans
    both majors (ADR-057). Coercing every value to a native scalar with ``.item()`` before serializing
    makes the digest identical across majors; on pandas 2 it reproduces the digest the existing
    bundled artifacts already store (verified byte-identical: ``aa21057f...``), so the fix is
    backward-compatible with NO metadata regeneration.

    NOTE: the shared ``_chirality.chirality_fingerprint`` carries the same fragile pattern for the
    other bundled models (xS / xCross / ghost-GK); that is a pre-existing cross-cutting issue tracked
    separately, not touched here.
    """

    def _native(v: object) -> object:
        # A missing value has THREE representations across pandas majors -- the probe's ball row
        # carries team_id/player_id = None, which stays None on pandas 2 (JSON ``null``) but coerces
        # to a float NaN on pandas 3 (JSON ``NaN``); nullable columns can also yield pd.NA. Collapse
        # all of them to a single canonical None FIRST, so the digest is identical across majors
        # (None-vs-NaN is the gap .item() alone does not bridge -- the r4 review finding). The values
        # come from ``to_dict("records")`` so each is a scalar; the guard is belt-and-braces against
        # pd.isna returning an array (which would make ``bool`` raise).
        if v is None or v is pd.NA:
            return None
        # float NaN covers both native ``float('nan')`` and ``np.float64('nan')`` (np.float64 subclasses
        # float), which is what a None id becomes on pandas 3. Guarded by isinstance so non-numeric
        # values (the string ids) never reach math.isnan.
        if isinstance(v, float) and math.isnan(v):
            return None
        # numpy scalars carry .item() -> native Python; native scalars pass through unchanged.
        item = getattr(v, "item", None)
        return item() if callable(item) else v

    records = [{k: _native(v) for k, v in rec.items()} for rec in frame.to_dict("records")]
    return hashlib.sha256(json.dumps(records, sort_keys=True, default=str).encode()).hexdigest()


def _outfield_chirality_block(model: GhostOutfieldModel) -> dict:
    """Behavioral chirality fingerprint (ADR-037) on the outfield probe: extractor + served mean.

    Built directly (not via ``chirality_fingerprint``, which hardcodes the ghost-GK canonical frame),
    so it runs on the outfield-shaped probe; the load-time check reuses the frame-agnostic
    ``verify_chirality``. The frame hash goes through :func:`_canonical_frame_sha` so it is
    pandas-major-invariant (ADR-057).
    """
    frame = _outfield_probe_frame()
    frame_sha = _canonical_frame_sha(frame)
    feats = _extract_all_ghost_outfield_features(
        frame, _outfield_probe_actions(), home_team_id="A", feature_set=model.feature_set
    )
    outputs = np.asarray(model.predict_mean(feats), dtype=float).ravel()
    if not np.all(np.isfinite(outputs)):
        raise ValueError(f"outfield chirality fingerprint produced non-finite outputs: {outputs!r}")
    return {
        "version": "outfield-chirality-1",
        "frame_sha256": frame_sha,
        "outputs": [round(float(v), 10) for v in outputs],
    }


def _outfield_feature_contract_block(feature_set: GhostOutfieldFeatureSet = "faithful") -> dict:
    """Feature contract (ADR-050) on the outfield probe: slot-1's feature vector + declared constants.

    The extractor consumes only pitch dimensions (field_length/width; the defensive-third boundary is
    length/3) --- all covered by the ``pitch_length``/``pitch_width`` fail-closed guard --- so
    ``constants`` is empty and the fingerprint (the feature vector itself) is the guard. Built directly
    rather than via ``feature_contract``, which hardcodes the ghost-GK contract frame.
    """
    frame = _outfield_probe_frame()
    probe_sha = _canonical_frame_sha(frame)  # pandas-major-invariant (ADR-057), same as the chirality block
    feats = _extract_all_ghost_outfield_features(
        frame, _outfield_probe_actions(), home_team_id="A", feature_set=feature_set
    )
    names = _feature_names_for(feature_set)
    vec = feats.sort_values("slot_index")[names].iloc[0].to_numpy(dtype=float)
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"outfield feature contract produced non-finite values: {vec!r}")
    return {
        "version": "outfield-feature-contract-1",
        "probe_sha256": probe_sha,
        "fingerprint": [round(float(v), 10) for v in vec],
        "constants": {},
    }


class GhostOutfieldModel:
    """League-average rest-defense rearguard-positioning model (TF-60 PR5).

    Mirrors :class:`~silly_kicks.tracking._ghost_gk.GhostGkModel`: two
    ``HistGradientBoostingRegressor`` ensembles (x, y) served as the exact pickle-free boosted mean
    (``baseline + sum_trees leaf_value``; no sklearn at inference, numba-accelerated leaf walk when the
    ``[numba]`` extra is installed). Parameters-only npz + JSON + SHA256 serialization; chirality +
    feature-contract fail-closed load-guards. No grid, no KDE density.

    Examples
    --------
    Fit on linked tracking frames + SPADL actions, then serve ghost rearguard positions::

        model = GhostOutfieldModel().fit(frames, actions, home_team_id=home_id)
        ghosts = serve_ghost_outfield_positions(frames, model=model, home_team_id=home_id)

    See NOTICE for full bibliographic citations.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 500,
        max_depth: int = 8,
        verbose: int = 0,
        feature_set: GhostOutfieldFeatureSet = "faithful",
    ):
        self.feature_set: GhostOutfieldFeatureSet = feature_set
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._verbose = verbose
        self._tree_nodes: list[np.ndarray] | None = None
        self._tree_nodes_y: list[np.ndarray] | None = None
        # Derived state (numba leaf kernels) -- rebuilt from _tree_nodes at fit/load, never serialized.
        self._flat_trees = None
        self._flat_trees_y = None
        self._baseline_x: float | None = None
        self._baseline_y: float | None = None
        self._sklearn_version: str | None = None
        self.corpus_provenance: dict | None = None
        self.training_commit: str | None = None
        self.training_platform: str | None = None
        # Transient sklearn regressors kept after fit() for the parity gate only (never serialized).
        self._sk_reg_x = None
        self._sk_reg_y = None

    def _feature_names(self) -> list[str]:
        return _feature_names_for(self.feature_set)

    def fit(
        self,
        frames: pd.DataFrame,
        actions: pd.DataFrame | None,
        *,
        home_team_id: int | str,
        carrier: pd.DataFrame | None = None,
        n_rearguard: int = 4,
    ) -> GhostOutfieldModel:
        """Train the two boosted-mean ensembles on per-(frame, slot) features + goal-relative targets.

        Examples
        --------
        Fit on a match's tracking frames and SPADL actions::

            model = GhostOutfieldModel(n_estimators=500, max_depth=8)
            model.fit(frames, actions, home_team_id=home_id)
        """
        data = _extract_all_ghost_outfield_features(
            frames,
            actions,
            home_team_id=home_team_id,
            carrier=carrier,
            feature_set=self.feature_set,
            n_rearguard=n_rearguard,
            both_teams=True,  # possession-conditioned training: model BOTH teams' deepest-n
        )
        if len(data) == 0:
            raise ValueError("ghost-outfield fit: the extractor produced no training rows.")
        return self._fit_extracted(data)

    def _fit_extracted(self, data: pd.DataFrame) -> GhostOutfieldModel:
        """Fit the two ensembles on a pre-extracted feature+target table (features + target_x/y).

        The trainer's cross-validation re-fits per fold on already-extracted rows, so extraction and
        the boosted fit are split: :meth:`fit` extracts then delegates here.
        """
        from sklearn.ensemble import HistGradientBoostingRegressor

        X = data[self._feature_names()].to_numpy(dtype=np.float64)
        y_x = data["target_x"].to_numpy(dtype=np.float64)
        y_y = data["target_y"].to_numpy(dtype=np.float64)

        def _make() -> HistGradientBoostingRegressor:
            return HistGradientBoostingRegressor(
                max_iter=self._n_estimators,
                max_depth=self._max_depth,
                categorical_features=None,  # type: ignore[arg-type]
                random_state=42,
                verbose=self._verbose,
            )

        reg_x = _make()
        reg_x.fit(X, y_x)
        reg_y = _make()
        reg_y.fit(X, y_y)

        for reg in (reg_x, reg_y):
            if not hasattr(reg, "_predictors") or reg._baseline_prediction.size != 1:
                raise RuntimeError(
                    "sklearn HistGradientBoostingRegressor private API changed -- reconstruction needs review"
                )

        self._tree_nodes = [tl[0].nodes.copy() for tl in reg_x._predictors]
        self._tree_nodes_y = [tl[0].nodes.copy() for tl in reg_y._predictors]
        self._flat_trees = _flatten_trees(self._tree_nodes)
        self._flat_trees_y = _flatten_trees(self._tree_nodes_y)
        self._baseline_x = float(reg_x._baseline_prediction.item())
        self._baseline_y = float(reg_y._baseline_prediction.item())
        self._sk_reg_x = reg_x
        self._sk_reg_y = reg_y
        return self

    def predict_mean(self, features: pd.DataFrame) -> np.ndarray:
        """Served estimate: the exact boosted HGBR mean, pickle-free + load-safe. Shape (n, 2).

        Examples
        --------
        Score a feature matrix produced by the extractor::

            positions = model.predict_mean(features)  # shape (n_rows, 2)
        """
        if (
            self._tree_nodes is None
            or self._tree_nodes_y is None
            or self._baseline_x is None
            or self._baseline_y is None
        ):
            raise RuntimeError("Model not fitted. Call .fit() or .load() first.")
        X = features[self._feature_names()].to_numpy(dtype=np.float64)
        out = np.empty((len(X), 2), dtype=np.float64)
        out[:, 0] = self._baseline_x + _vectorized_leaf_values(self._tree_nodes, X, flat=self._flat_trees)
        out[:, 1] = self._baseline_y + _vectorized_leaf_values(self._tree_nodes_y, X, flat=self._flat_trees_y)
        return out

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Alias for :meth:`predict_mean` (the served estimate).

        Examples
        --------
        ::

            positions = model.predict(features)
        """
        return self.predict_mean(features)

    def save(self, path: Path) -> None:
        """Serialize to model.npz + metadata.json + SHA256SUMS (no pickle; parameters only).

        Examples
        --------
        Serialize a fitted model to a directory::

            model.save(Path("models/ghost_outfield_v1/default"))
        """
        if (
            self._tree_nodes is None
            or self._tree_nodes_y is None
            or self._baseline_x is None
            or self._baseline_y is None
        ):
            raise RuntimeError("Model not fitted. Call .fit() first.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        save_dict: dict[str, np.ndarray] = {
            "n_trees": np.array([len(self._tree_nodes)]),
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
        np.savez_compressed(str(path / "model.npz"), **save_dict)  # type: ignore[arg-type]

        import sklearn

        metadata = {
            "feature_names": self._feature_names(),
            "n_estimators": self._n_estimators,
            "max_depth": self._max_depth,
            "sklearn_version": self._sklearn_version or sklearn.__version__,
            "training_commit": self.training_commit,
            "training_platform": self.training_platform,
            "corpus_provenance": self.corpus_provenance,
            "serve_estimator": SERVED_ESTIMATOR,
            "version": "1.0.0",
            "stores_training_data": False,
            "pitch_length": _geo.PITCH_LENGTH,
            "pitch_width": _geo.PITCH_WIDTH,
            "feature_set": self.feature_set,
            "chirality": _outfield_chirality_block(self),
            "feature_contract": _outfield_feature_contract_block(self.feature_set),
        }
        with open(path / "metadata.json", "w", newline="\n") as f:
            json.dump(metadata, f, indent=2)

        with open(path / "SHA256SUMS", "w", newline="\n") as f:
            for fname in ["model.npz", "metadata.json"]:
                raw = (path / fname).read_bytes()
                if fname.endswith(".json"):
                    raw = raw.replace(b"\r\n", b"\n")
                f.write(f"{hashlib.sha256(raw).hexdigest()}  {fname}\n")

    @classmethod
    def load(cls, path: Path, *, legacy_override: bool = False) -> GhostOutfieldModel:
        """Load from a local directory with SHA-256 + chirality + feature-contract verification.

        Examples
        --------
        Load a previously saved model from disk::

            model = GhostOutfieldModel.load(Path("models/ghost_outfield_v1/default"))
        """
        path = Path(path)

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
                if fname.endswith(".json"):
                    raw = raw.replace(b"\r\n", b"\n")
                actual_hash = hashlib.sha256(raw).hexdigest()
                if actual_hash != expected_hash:
                    raise IntegrityError(
                        f"Integrity check failed for {fname}: expected {expected_hash}, got {actual_hash}"
                    )

        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        recorded_estimator = metadata.get("serve_estimator", SERVED_ESTIMATOR)
        if recorded_estimator != SERVED_ESTIMATOR:
            raise IntegrityError(
                f"Model metadata serve_estimator={recorded_estimator!r} != code "
                f"SERVED_ESTIMATOR={SERVED_ESTIMATOR!r}; refusing to serve a mismatched estimator."
            )

        def _load_ensemble(data, nodes_prefix: str, dtype_prefix: str, count_key: str) -> list[np.ndarray]:
            nodes_list = []
            for i in range(int(data[count_key][0])):
                raw_bytes = np.array(data[f"{nodes_prefix}{i}"])
                dtype = np.dtype(ast.literal_eval(str(data[f"{dtype_prefix}{i}"][0])))
                nodes_list.append(raw_bytes.view(dtype))
            return nodes_list

        with np.load(path / "model.npz", allow_pickle=False) as data:
            tree_nodes = _load_ensemble(data, "tree_nodes_", "tree_dtype_", "n_trees")
            tree_nodes_y = _load_ensemble(data, "tree_nodes_y_", "tree_dtype_y_", "n_trees_y")
            baseline_x = float(data["baseline_x"][0])
            baseline_y = float(data["baseline_y"][0])

        model = cls(
            n_estimators=metadata.get("n_estimators", 500),
            max_depth=metadata.get("max_depth", 8),
            feature_set=metadata.get("feature_set", "faithful"),
        )
        model._tree_nodes = tree_nodes
        model._tree_nodes_y = tree_nodes_y
        model._flat_trees = _flatten_trees(tree_nodes)
        model._flat_trees_y = _flatten_trees(tree_nodes_y)
        model._baseline_x = baseline_x
        model._baseline_y = baseline_y
        model._sklearn_version = metadata.get("sklearn_version")
        model.training_commit = metadata.get("training_commit")
        model.training_platform = metadata.get("training_platform")
        model.corpus_provenance = metadata.get("corpus_provenance")

        # Pitch-dimension guard (fail-closed): a dimension change skews every goal-relative feature.
        rec_len = metadata.get("pitch_length")
        rec_wid = metadata.get("pitch_width")
        if rec_len is not None and (rec_len != _geo.PITCH_LENGTH or rec_wid != _geo.PITCH_WIDTH):
            raise IntegrityError(
                f"Pitch-dimension mismatch: model trained on {rec_len}x{rec_wid} m, library is "
                f"{_geo.PITCH_LENGTH}x{_geo.PITCH_WIDTH} m. Goal-relative features would be skewed; "
                "refusing to load (retrain required)."
            )

        verify_chirality(
            _outfield_chirality_block(model),
            metadata.get("chirality"),
            legacy_override=legacy_override,
            model_name="ghost-outfield",
            error_cls=IntegrityError,
        )
        verify_feature_contract(
            _outfield_feature_contract_block(model.feature_set),
            metadata.get("feature_contract"),
            legacy_override=legacy_override,
            model_name="ghost-outfield",
            error_cls=IntegrityError,
        )
        return model

    @classmethod
    def from_variant(
        cls, name: GhostOutfieldVariant = "default", *, legacy_override: bool = False
    ) -> GhostOutfieldModel:
        """Load a bundled variant from ``_ghost_outfield_weights/<name>/`` (or ``$SILLY_KICKS_GHOST_OUTFIELD_PATH``).

        Examples
        --------
        Load the bundled default variant::

            model = GhostOutfieldModel.from_variant("default")
        """
        root = Path(os.environ.get(_ENV_VAR, _WEIGHTS_ROOT))
        return cls.load(root / name, legacy_override=legacy_override)

    @classmethod
    def from_hub(
        cls,
        repo_id: str = _HF_REPO_ID,
        *,
        revision: str | None = None,
        legacy_override: bool = False,
    ) -> GhostOutfieldModel:
        """Download a published variant from the Hugging Face Hub and load it (fail-closed).

        Examples
        --------
        Download and load the published default variant::

            model = GhostOutfieldModel.from_hub("silly-kicks/ghost-outfield-v1")
        """
        from huggingface_hub import snapshot_download

        local = snapshot_download(repo_id=repo_id, revision=revision)
        return cls.load(Path(local), legacy_override=legacy_override)


# ---------------------------------------------------------------------------
# Serve seam (the tracking public surface PR6 consumes)
# ---------------------------------------------------------------------------

#: Closed provenance vocabulary for the serve output's ``ghost_outfield_source`` column.
#: ``fov_cropped`` = the rearguard region was not sufficiently observed on an SB360 freeze frame, so
#: the deepest-n VISIBLE players would be a promoted (fabricated) rearguard -> honest-NaN (spec 8).
GHOST_OUTFIELD_SOURCE_VALUES: frozenset[str] = frozenset({"computed", "variant_unavailable", "fov_cropped"})

#: The rearguard region (A's defensive third) must be at least this fraction observed for the visible
#: deepest-n selection to be trustworthy; below it a deeper defender could be cropped out of the FOV and
#: a midfielder promoted into a slot, so the frame's ghosts are honest-NaN (ADR-077 / spec 8).
_RD_FOV_MIN_OBSERVED: float = 0.9

_SERVE_OUTPUT_COLS: list[str] = [
    "game_id",
    "period_id",
    "frame_id",
    "team_id",
    "slot_index",
    "player_id",
    "ghost_gr_x",
    "ghost_gr_y",
    "ghost_outfield_source",
]


def _resolve_outfield_model_for_frames(
    frames: pd.DataFrame, model: GhostOutfieldModel | GhostOutfieldVariant | None
) -> tuple[GhostOutfieldModel | None, str]:
    """Velocity-keyed variant auto-select (ADR-067).

    A supplied :class:`GhostOutfieldModel` is used verbatim (``"custom"``). Otherwise the variant is
    keyed off the frames' declared velocity availability (``"default"`` on velocity-bearing frames,
    ``"position_only"`` on declared-velocity-less freeze frames), or taken from an explicit variant
    string. A variant that is not bundled resolves to ``None`` -> the serve emits honest-NaN rows;
    crucially a missing ``position_only`` is **never** silently replaced by ``default`` (the ADR-067
    asymmetry: the default velocity model is invalid on velocity-less frames), because this resolver
    never falls back to a different key.
    """
    if isinstance(model, GhostOutfieldModel):
        return model, "custom"
    base = model if isinstance(model, str) else _variant_key_for_velocity(frames)
    root = Path(os.environ.get(_ENV_VAR, _WEIGHTS_ROOT))
    if not (root / base / "SHA256SUMS").exists():
        # Variant is NOT bundled -> honest-NaN (ADR-067). A missing position_only is never replaced
        # by default (the resolver never falls back to a different key).
        return None, base
    # The artifact IS present: a SHA-256 / chirality / feature-contract failure is CORRUPTION, not
    # "unavailable" -- let IntegrityError (and any FileNotFoundError from a truncated bundle)
    # PROPAGATE rather than silently serving an all-NaN "variant_unavailable" column that would hide
    # a bad artifact (e.g. a pandas-major chirality skew) from every consumer (IMPL-06b).
    return GhostOutfieldModel.from_variant(base), base  # type: ignore[arg-type]


def _rearguard_region_ltr(goal_x: float) -> np.ndarray:
    """A's defensive third as a convex rectangle in the LTR frame (the rearguard's home region)."""
    if goal_x < 50.0:  # A defends the left goal
        return np.array(
            [[0.0, 0.0], [_DEF_THIRD_GR_X, 0.0], [_DEF_THIRD_GR_X, _FIELD_WIDTH], [0.0, _FIELD_WIDTH]],
            dtype=float,
        )
    return np.array(  # A defends the right goal
        [
            [_FIELD_LENGTH - _DEF_THIRD_GR_X, 0.0],
            [_FIELD_LENGTH, 0.0],
            [_FIELD_LENGTH, _FIELD_WIDTH],
            [_FIELD_LENGTH - _DEF_THIRD_GR_X, _FIELD_WIDTH],
        ],
        dtype=float,
    )


def _apply_fov_cropping(out: pd.DataFrame, frames: pd.DataFrame, visible_area: pd.DataFrame) -> None:
    """Honest-NaN the served rows on SB360 frames whose rearguard region is FOV-cropped (spec 8, IN PLACE).

    ``visible_area`` is a per-frame polygon table (``game_id, period_id, frame_id, visible_area``), the
    ``snapshot_to_tracking_frames`` FOV. For each served ``(frame, team A)`` we measure the observed
    fraction of A's defensive third against that frame's polygon; when it falls below
    :data:`_RD_FOV_MIN_OBSERVED` (or no polygon is present) the deepest-n VISIBLE players are an
    unreliable rearguard --- a deeper defender could be cropped and a midfielder promoted into a slot ---
    so the ghosts are set to NaN and tagged ``fov_cropped`` rather than served as ``computed``.
    """
    goal_map = resolve_defended_goals(frames)
    # ONE polygon lookup per frame, canonical-keyed (ADR-019 / ADR-055 rule 2); built once.
    poly_by_frame: dict = {}
    for _, row in visible_area.iterrows():
        poly = row.get("visible_area")
        if poly is None:
            continue
        poly_by_frame[(canonical_id(row["game_id"]), row["period_id"], row["frame_id"])] = as_polygon(poly)

    # Per (frame, team) crop verdict, computed once over the unique keys (no per-row rescan; ADR-068).
    keys = out[["game_id", "period_id", "frame_id", "team_id"]].drop_duplicates()
    cropped_frames: set = set()
    for _, k in keys.iterrows():
        gid, pid, fid, a_team = k["game_id"], k["period_id"], k["frame_id"], k["team_id"]
        goal_x = goal_map.get(gid, pid, a_team, allow_guess=True)
        poly = poly_by_frame.get((canonical_id(gid), pid, fid))
        if goal_x is None or poly is None:
            cropped_frames.add((canonical_id(gid), pid, fid))
            continue
        frac = region_observed_fraction(poly, _rearguard_region_ltr(goal_x))
        if not np.isfinite(frac) or frac < _RD_FOV_MIN_OBSERVED:
            cropped_frames.add((canonical_id(gid), pid, fid))

    if not cropped_frames:
        return
    row_keys = list(zip(out["game_id"].map(canonical_id), out["period_id"], out["frame_id"], strict=True))
    mask = np.array([rk in cropped_frames for rk in row_keys], dtype=bool)
    out.loc[mask, ["ghost_gr_x", "ghost_gr_y"]] = np.nan
    out.loc[mask, "ghost_outfield_source"] = "fov_cropped"


def serve_ghost_outfield_positions(
    frames: pd.DataFrame,
    *,
    model: GhostOutfieldModel | GhostOutfieldVariant | None = None,
    home_team_id: int | str,
    actions: pd.DataFrame | None = None,
    carrier: pd.DataFrame | None = None,
    n_rearguard: int = 4,
    visible_area: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Serve league-average ghost rearguard positions --- the ``tracking`` seam PR6 consumes.

    Returns one row per ``(game_id, period_id, frame_id, team_id, slot_index)`` with **goal-relative**
    ``ghost_gr_x`` / ``ghost_gr_y`` (the caller writes back to frame coordinates, mirroring
    :func:`serve_ghost_gk_positions`), the bookkeeping key ``player_id`` (the actual player in the
    slot, for the PR6 actual<->ghost match), and a ``ghost_outfield_source`` provenance token from
    :data:`GHOST_OUTFIELD_SOURCE_VALUES`.

    Velocity contract (ADR-054/063): a MIXED-availability frame set raises; an UNDECLARED missing
    ``vx``/``vy`` (a forgotten ``derive_velocities()``) raises; a declared-velocity-less freeze frame
    auto-selects the ``position_only`` variant (ADR-067). A variant that is not bundled yields
    honest-NaN ghost positions (``ghost_outfield_source="variant_unavailable"``), never a fabricated
    coordinate and never a fall-back to the invalid default.

    FOV contract (SB360; spec 8, ADR-077): pass ``visible_area`` (a per-frame polygon table
    ``game_id, period_id, frame_id, visible_area``) on freeze-frame input. Any frame whose rearguard
    region (A's defensive third) is not sufficiently observed is served honest-NaN
    (``ghost_outfield_source="fov_cropped"``) --- otherwise the deepest-n VISIBLE players would be a
    fabricated rearguard (a cropped-out deep defender silently replaced by a promoted midfielder). With
    no ``visible_area`` (full-coverage tracking) every scored frame is ``computed``.
    """
    if len(frames) == 0:
        return pd.DataFrame(columns=_SERVE_OUTPUT_COLS)

    if _velocity_availability_is_mixed(frames):
        raise ValueError(
            "serve_ghost_outfield_positions: mixed velocity-availability -- some rows declare "
            "speed_source unavailable and some do not. Pass a single-availability frame set."
        )
    if not _velocity_unavailable_by_design(frames) and ("vx" not in frames.columns or "vy" not in frames.columns):
        raise ValueError(
            "serve_ghost_outfield_positions requires vx/vy on frames (call derive_velocities() "
            "first), or declare speed_source unavailable. See the velocity-availability contract."
        )

    resolved, key = _resolve_outfield_model_for_frames(frames, model)
    feature_set: GhostOutfieldFeatureSet = (
        resolved.feature_set if resolved is not None else ("position_only" if key == "position_only" else "faithful")
    )
    # Resolve possession like the ghost-GK serve does: if the caller supplied neither a carrier nor a
    # team_in_possession column, infer the ball carrier on the full frames (the in-possession team is
    # the rearguard's owner). PR6 passes raw frames, so this keeps the seam usable on them.
    if carrier is None and "team_in_possession" not in frames.columns:
        from ._ball_carrier import DEFAULT_CARRIER_PARAMS, infer_ball_carrier

        carrier_params: dict = dict(DEFAULT_CARRIER_PARAMS)
        carrier = infer_ball_carrier(frames, **carrier_params)[
            ["game_id", "period_id", "frame_id", "ball_carrier_team_id"]
        ]
    data = _extract_all_ghost_outfield_features(
        frames,
        actions,
        home_team_id=home_team_id,
        carrier=carrier,
        feature_set=feature_set,
        n_rearguard=n_rearguard,
    )
    key_cols = ["game_id", "period_id", "frame_id", "team_id", "slot_index", "player_id"]
    out = data[key_cols].copy() if len(data) else pd.DataFrame(columns=key_cols)

    if resolved is None:
        out["ghost_gr_x"] = np.nan
        out["ghost_gr_y"] = np.nan
        out["ghost_outfield_source"] = "variant_unavailable"
    else:
        if len(data):
            preds = resolved.predict_mean(data)
            out["ghost_gr_x"] = preds[:, 0]
            out["ghost_gr_y"] = preds[:, 1]
        else:
            out["ghost_gr_x"] = pd.Series(dtype=float)
            out["ghost_gr_y"] = pd.Series(dtype=float)
        out["ghost_outfield_source"] = "computed"
        # FOV honest-NaN (spec 8): only meaningful on computed rows -- a cropped rearguard region
        # means the deepest-n VISIBLE selection is untrustworthy, so override those ghosts to NaN.
        if visible_area is not None and len(out):
            _apply_fov_cropping(out, frames, visible_area)
    return out[_SERVE_OUTPUT_COLS]


# ---------------------------------------------------------------------------
# Ghost-rearguard coherence (spec 9: measure, then escalate to a shape constraint IF incoherent)
# ---------------------------------------------------------------------------


def ghost_rearguard_coherence(served: pd.DataFrame) -> dict:
    """Measure whether an independently-predicted ghost rearguard hangs together as a line.

    ``served`` is a :func:`serve_ghost_outfield_positions` output. Reported (spec 9), NOT gated: the
    per-slot model is built independently, and a shape constraint is added only if this measurement
    shows material incoherence. Two metrics per ``(game, period, frame, team)`` group of >= 2 finite
    ghost slots:

    * ``ordering_fraction`` --- the fraction of groups whose ghost slots preserve their lateral rank
      (``ghost_gr_y`` non-decreasing in ``slot_index`` order). 1.0 = every ghost line is well ordered.
    * ``min_pairwise_distance_m`` --- the smallest slot-to-slot distance seen across all groups (a
      collapse indicator).
    """
    finite = served.dropna(subset=["ghost_gr_x", "ghost_gr_y"])
    n_groups = 0
    n_ordered = 0
    min_dists: list[float] = []
    for _key, grp in finite.groupby(["game_id", "period_id", "frame_id", "team_id"], sort=False):
        gg = grp.sort_values("slot_index", kind="stable")
        if len(gg) < 2:
            continue
        n_groups += 1
        ys = gg["ghost_gr_y"].to_numpy(dtype=float)
        xs = gg["ghost_gr_x"].to_numpy(dtype=float)
        if np.all(np.diff(ys) >= 0.0):
            n_ordered += 1
        pts = np.column_stack([xs, ys])
        dmat = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=-1))
        np.fill_diagonal(dmat, np.inf)
        min_dists.append(float(dmat.min()))
    return {
        "n_groups": n_groups,
        "ordering_fraction": (n_ordered / n_groups) if n_groups else float("nan"),
        "min_pairwise_distance_m": float(np.min(min_dists)) if min_dists else float("nan"),
    }
