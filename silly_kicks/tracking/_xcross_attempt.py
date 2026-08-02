"""TF-17 xCrossAttempt: per-frame cross-attempt-propensity model (GKDV Layer 2).

Cross analogue of xShotOccurrence (TF-16). STATE-anchored occurrence surface:
P(the in-possession team attempts a cross within ~horizon of a frame). Inspired by
Cao et al. (2025, arXiv:2505.11841); extended with goalkeeper-position confounders.
See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from silly_kicks._nan_safety import nan_safe_enrichment
from silly_kicks.id_compat import align_join_keys, canonical_id, canonical_id_series, same_id
from silly_kicks.spadl import config as _spc
from silly_kicks.tracking import _geometry as _geo
from silly_kicks.tracking._ball_carrier import (
    DEFAULT_CARRIER_PARAMS,
    derive_team_in_possession,
    infer_ball_carrier,
)
from silly_kicks.tracking._occurrence_labels import _build_occurrence_labels
from silly_kicks.tracking._xshot_occurrence import IntegrityError, load_xgb_booster_base_score_safe
from silly_kicks.tracking.utils import link_actions_to_frames

XCrossFeatureSet = Literal["faithful", "extended"]

# Paper confounders (realized with silly-kicks primitives) + ball geometry.
# NOTE (review C1): goal-relative convention is **attacked goal at gr_x = 0**
# (verified: _geometry.to_goal_relative_x(95, goal_x=105) == 10.0). Every distance-to-goal /
# box / post formula below measures from gr_x = 0, NOT PITCH_LENGTH.
# NOTE (review H2): paper confounder #7 "crosser position (FW/MF/DF)" is DROPPED in PR-A -- a
# tracking longitudinal-role proxy == the carrier's gr_x == `dist_endline`, i.e. collinear with
# #5 (zero added signal). The paper's #7 is categorical event metadata; a faithful proxy needs
# season/role aggregation not available in PR-A. Documented in NOTICE; candidate for `extended`.
_BALL_FEATURES = ["ball_r", "ball_theta", "ball_speed"]
_CONFOUNDERS = [
    "score_differential",  # #1 (NaN if no score lookup supplied)
    "dist_nearest_def",  # #2 carrier -> nearest opponent
    "space_controlled",  # #3 Voronoi dominant-region area proxy (cache-free)
    "dist_nearest_teammate",  # #4 carrier -> nearest teammate
    "dist_endline",  # #5 carrier -> attacked goal line (== carrier gr_x)
    "box_off_def_ratio",  # #6 attackers/defenders inside the attacked penalty box
    "ten_minute_warning",  # #8 final 10 min of the half (0/1)
]
# The NOVEL, contiguous, separately-droppable GK block (the headline extension).
XCROSS_GK_BLOCK = [
    "gk_r",
    "gk_theta",
    "gk_lateral_offset",
    "gk_dist_near_post",
    "gk_dist_far_post",
    "gk_carrier_side",
]
XCROSS_FEATURE_NAMES_FAITHFUL = _BALL_FEATURES + _CONFOUNDERS + XCROSS_GK_BLOCK  # 16 (3+7+6)

# Domain-filter constants (M1 re-justified; M-4 corridor = cross_zone dense-zone, not SkillCorner's).
_WIDE_Y_LOW = 14.0  # cross_zone wide corridor (specialty.py:54)
_WIDE_Y_HIGH = 54.0
_ADVANCE_M = 35.0  # permissive default (SkillCorner _is_cross start_x>70 == 35 m); PR-B re-selects
# Canonical penalty-area geometry. Aliased rather than inlined at the `in_box` use site below:
# that predicate is VECTORIZED over numpy arrays, so it cannot call the scalar
# `_geometry.in_penalty_area_goal_relative` without a per-element loop -- the single source here
# is the CONSTANT, not the predicate. Keeping the names also keeps them visible to the geometry
# constant enumeration gate, which reads module-level assignments.
_BOX_DEPTH_M = _spc.penalty_area_depth  # 16.5
_BOX_HALF_WIDTH_M = _spc.penalty_area_half_width  # 20.16 (40.32/2)
_GOAL_HALF_WIDTH_M = 3.66  # goal width 7.32 / 2

_DEFAULT_CROSS_TYPES = ("cross",)  # open-play only (corner/freekick excluded by default)
_DEFAULT_CARRIER_PARAMS = DEFAULT_CARRIER_PARAMS

_HF_REPO_ID = "silly-kicks/xcross-attempt-v1"
_MODEL_VERSION = "1.0.0"
_XCROSS_WEIGHTS_ROOT = Path(__file__).parent / "_xcross_weights"
_VARIANT_CACHE: dict = {}
# See _xshot_occurrence.py for the full rationale: "public" is a stale alias that fell through
# to the Hub-hosted RESTRICTED sc_extended artifact; the bundled "default" IS the public arm.
# Resolve the alias BEFORE the cache (spec 2026-07-20 §8).
_VARIANT_ALIASES = {"public": "default"}
_HUB_VARIANTS = frozenset({"sc_extended"})
_INT_PARAMS = ("n_estimators", "max_depth", "min_child_weight")


def _polar(dx: float, dy: float) -> tuple[float, float]:
    return math.hypot(dx, dy), math.atan2(dy, dx)


def _nearest_dist(origin_xy, pts_xy) -> float:
    if not pts_xy:
        return np.nan
    ox, oy = origin_xy
    return float(min(math.hypot(px - ox, py - oy) for px, py in pts_xy))


def _grid_centres(length: float, res: float) -> np.ndarray:
    """Cell centres tiling ``length`` symmetrically about its midpoint.

    ``a = L/2 - (n-1)*res/2`` yields 1.5 for (105, 3) -- byte-identical to the shipped x grid --
    and 1.0 for (68, 3), and stays mirror-symmetric for ANY ``res``. The previous
    ``arange(res/2, L, res)`` is symmetric only when ``L`` divides evenly by ``res``: true for
    105/3, FALSE for 68/3, which centred the y grid on 34.5 instead of 34.0. A scene and its
    left-right mirror at the SAME goal end then differed by 5.4% in ``space_controlled`` --
    xCross model feature #3, on the left-wing/right-wing axis, in a CROSS model.

    Do not "simplify" back to ``res/2``: the failure inverts with ``res``. At ``res=2.0`` it is the
    X grid that becomes asymmetric (centres 1, 3, ..., 103; ``105 - 1 = 104`` is not a centre), so
    a res-specific comment would be actively misleading -- and ``res`` has already moved once.
    """
    n = round(length / res)
    anchor = length / 2.0 - (n - 1) * res / 2.0
    return anchor + res * np.arange(n)


def _dominant_region_area(carrier_xy, all_xy, *, res: float = 3.0) -> float:
    """Cache-free 'space controlled' proxy: fraction of pitch grid cells whose nearest player is
    the carrier x pitch area. numpy-only nearest-player Voronoi approximation; NO pitch-control
    cache (locks the TF-19 counterfactual guarantee).

    PA-L2 perf note: builds a full-pitch meshgrid PER FRAME (~805 cells x ~22 players at res=3.0;
    35 x 23). The former "~1800" was stale -- it describes res=2.0 (52 x 34 = 1768) and went out of
    date when ``res`` moved to 3.0.
    The perf guard is structural (call-count), not wall-clock; PR-B must add a real-data wall-clock
    sanity check and coarsen ``res`` if this dominates. It is the only non-vectorized feature.
    """
    if carrier_xy is None or not all_xy:
        return np.nan
    xs = _grid_centres(_geo.PITCH_LENGTH, res)
    ys = _grid_centres(_geo.PITCH_WIDTH, res)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.asarray(all_xy, dtype=float)  # (P, 2)
    d2 = (gx[..., None] - pts[:, 0]) ** 2 + (gy[..., None] - pts[:, 1]) ** 2
    nearest = d2.argmin(axis=-1)
    carrier_idx = int(np.argmin((pts[:, 0] - carrier_xy[0]) ** 2 + (pts[:, 1] - carrier_xy[1]) ** 2))
    frac = float((nearest == carrier_idx).mean())
    return frac * _geo.PITCH_LENGTH * _geo.PITCH_WIDTH


def extract_xcross_features(
    frame_data: pd.DataFrame,
    *,
    gk_team_id,
    goal_x: float,
    carrier_player_id,
    feature_set: XCrossFeatureSet = "faithful",
    score_differential: float = np.nan,
) -> pd.DataFrame:
    """Extract the faithful xCross feature row for a single (game, period, frame) snapshot.

    All spatial features are in goal-relative coordinates (attacked goal at gr_x = 0). The GK block
    is the contiguous tail (``XCROSS_GK_BLOCK``). ``score_differential`` is supplied by the caller
    (prepare/compute build it from match-context actions; NaN when unavailable). NaN-tolerant.
    """
    if feature_set != "faithful":
        raise NotImplementedError(
            "xCrossAttempt feature_set='extended' is a deferred extension point; "
            "only 'faithful' is implemented (TF-17 PR-A). See the design spec."
        )
    f = frame_data
    is_ball = f["is_ball"].to_numpy(dtype=bool)
    is_gk = f["is_goalkeeper"].to_numpy(dtype=bool)
    # CANONICAL id arrays (ADR-019): a nullable-Int64 id column (GradientSports) -> .to_numpy()
    # upcasts to float64, so a naive ``.astype(str)`` yields "366.0" and never matches the clean-int
    # carrier id "366" -> carrier_mask all-False -> every carrier-anchored feature (and the
    # carrier-gated GK block) silently NaN. canonical_id_series collapses 366/366.0/Int64(366)/"366"
    # -> "366"; the sentinel fill keeps NA (ball rows) out of the comparisons. Genuine-string
    # providers (kloppy) are unchanged, so the trained public model needs no retrain.
    na_fill = "\x00"  # sentinel: no canonical id equals it, so NA (ball) rows never match
    team = canonical_id_series(f["team_id"]).fillna(na_fill).to_numpy()
    pid = canonical_id_series(f["player_id"]).fillna(na_fill).to_numpy()
    gr_x = np.array([_geo.to_goal_relative_x(x, goal_x=goal_x) for x in f["x"].to_numpy()])
    # y is GOAL-RELATIVE from here down (PR 5). Paired with gr_x this is the 180-degree point
    # reflection; before PR 5 y rode through untransformed, so `atan2(y - GOAL_Y, gr_x)` negated
    # every bearing between the two goal ends while every radial stayed byte-identical. The local
    # name is kept because every consumer below already reads `y`.
    y = np.array([_geo.to_goal_relative_y(v, goal_x=goal_x) for v in f["y"].to_numpy(dtype=float)])

    out: dict[str, float] = {name: np.nan for name in XCROSS_FEATURE_NAMES_FAITHFUL}

    # Ball (ball-anchored)
    if is_ball.any():
        bx, by = float(gr_x[is_ball][0]), float(y[is_ball][0])
        out["ball_r"], out["ball_theta"] = _polar(bx, by - _geo.GOAL_Y)
        bvx = float(f.loc[is_ball, "vx"].to_numpy()[0])
        bvy = float(f.loc[is_ball, "vy"].to_numpy()[0])
        out["ball_speed"] = math.hypot(bvx, bvy)

    # Carrier-anchored geometry. Match the carrier by CANONICAL id (pid is already canonical above),
    # so it works whether the frame player_id is native string (kloppy/sportec/skillcorner/metrica)
    # OR nullable Int64 (gradientsports) -- see the canonical-id comment above.
    carrier_mask = np.zeros(len(f), dtype=bool)
    has_carrier = carrier_player_id is not None and not (
        isinstance(carrier_player_id, float) and math.isnan(carrier_player_id)
    )
    carrier_key = canonical_id(carrier_player_id) if has_carrier else None
    if has_carrier:
        carrier_mask = (pid == carrier_key) & ~is_ball
    cx = cy = None
    if carrier_mask.any():
        cx, cy = float(gr_x[carrier_mask][0]), float(y[carrier_mask][0])
        carrier_team = team[carrier_mask][0]
        opp = [(gr_x[i], y[i]) for i in range(len(f)) if not is_ball[i] and team[i] != carrier_team]
        mate = [
            (gr_x[i], y[i])
            for i in range(len(f))
            if not is_ball[i] and team[i] == carrier_team and pid[i] != carrier_key
        ]
        all_xy = [(gr_x[i], y[i]) for i in range(len(f)) if not is_ball[i]]
        out["dist_nearest_def"] = _nearest_dist((cx, cy), opp)
        out["dist_nearest_teammate"] = _nearest_dist((cx, cy), mate)
        out["dist_endline"] = float(cx)  # C1: attacked goal at gr_x=0 -> distance to endline IS cx
        out["space_controlled"] = _dominant_region_area((cx, cy), all_xy)

    # GK block (NOVEL extension) -- defending GK = is_goalkeeper row on gk_team_id.
    # C1: posts live at the ATTACKED goal line gr_x = 0 (NOT PITCH_LENGTH). Gated on carrier (needs cy).
    gk_mask = is_gk & (team == canonical_id(gk_team_id))
    if gk_mask.any() and cx is not None and cy is not None:  # cy set iff the carrier resolved
        gkx, gky = float(gr_x[gk_mask][0]), float(y[gk_mask][0])
        out["gk_r"], out["gk_theta"] = _polar(gkx, gky - _geo.GOAL_Y)
        out["gk_lateral_offset"] = float(gky - _geo.GOAL_Y)
        # carrier flank sign; `or 1.0` only guards the exactly-central carrier (cy == GOAL_Y).
        side = np.sign(cy - _geo.GOAL_Y) or 1.0
        near_post_y = _geo.GOAL_Y + _GOAL_HALF_WIDTH_M * side  # post on the carrier's flank
        far_post_y = _geo.GOAL_Y - _GOAL_HALF_WIDTH_M * side
        out["gk_dist_near_post"] = math.hypot(gkx, gky - near_post_y)  # goal at gr_x=0
        out["gk_dist_far_post"] = math.hypot(gkx, gky - far_post_y)
        out["gk_carrier_side"] = float((gky - _geo.GOAL_Y) * side)

    # #6 off/def ratio in the attacked box. C1: attacked box is goal-relative gr_x <= 16.5
    # (attacked goal at gr_x=0), |y-34| <= 20.16.
    in_box = (gr_x <= _BOX_DEPTH_M) & (np.abs(y - _geo.GOAL_Y) <= _BOX_HALF_WIDTH_M) & ~is_ball
    if carrier_mask.any():
        carrier_team = team[carrier_mask][0]
        n_off = int(((team == carrier_team) & in_box & ~is_gk).sum())
        n_def = int(((team != carrier_team) & in_box & ~is_gk).sum())
        out["box_off_def_ratio"] = float(n_off / n_def) if n_def > 0 else float(n_off)

    # #1 score differential (passed in; NaN at serve unless a score lookup supplied it)
    out["score_differential"] = (
        float(score_differential) if not (score_differential is None or math.isnan(score_differential)) else np.nan
    )

    # #8 ten-minute warning (final 10 min of a 45-min half). PA-M1: relies on time_seconds being
    # PER-PERIOD (resets to 0 each half) -- verified: the kloppy gateway sets
    # time_seconds = frame.timestamp.total_seconds() and kloppy timestamps are period-relative
    # (sportec native expects the same contract). ET periods (3/4/5) get 0 by design (the paper's
    # feature is "final 10 min of a half"). Locked by test_ten_minute_warning_period2_early_is_zero.
    t = float(f["time_seconds"].iloc[0])
    period = int(f["period_id"].iloc[0])
    out["ten_minute_warning"] = 1 if period in (1, 2) and t >= 35 * 60 else 0

    return pd.DataFrame([out], columns=XCROSS_FEATURE_NAMES_FAITHFUL)


def build_xcross_labels(
    frames_index: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    horizon_seconds: float = 1.0,
    cross_types: tuple[str, ...] = _DEFAULT_CROSS_TYPES,
    frame_team_col: str = "team_in_possession",
) -> pd.Series:
    """Per-frame xCross label: 1 iff a same-team cross occurs in [t, t+horizon] (same period).

    Open-play ``cross`` only by default; ``cross_types`` can add ``corner_crossed`` /
    ``freekick_crossed`` for sensitivity. Thin wrapper over the shared occurrence helper.
    """
    type_ids = {_spc.actiontype_id[t] for t in cross_types}
    crosses = actions[actions["type_id"].isin(type_ids)][["game_id", "period_id", "team_id", "time_seconds"]]
    y = _build_occurrence_labels(frames_index, crosses, horizon=horizon_seconds, frame_team_col=frame_team_col)
    return pd.Series(y, index=frames_index.index)


def _build_goal_map(frames: pd.DataFrame) -> dict:
    """Precompute (game_id, period_id, team_id) -> defended goal_x ONCE (H1; mirror xS
    _defended_goal_x at the frame-group level). Excludes the "ball" pseudo-team (PA-L3)."""
    goal_map: dict[tuple, float] = {}
    real = frames[frames["team_id"] != "ball"]  # PA-L3: skip the ball "team"
    gk = real[real["is_goalkeeper"]]
    for (gid, pid, tid), grp in real.groupby(["game_id", "period_id", "team_id"], sort=False):
        gk_grp = gk[(gk["game_id"] == gid) & (gk["period_id"] == pid) & (gk["team_id"] == tid)]
        mean_x = gk_grp["x"].mean() if len(gk_grp) else grp["x"].mean()
        goal_map[(gid, pid, tid)] = 0.0 if mean_x < _geo.PITCH_LENGTH / 2 else _geo.PITCH_LENGTH
    return goal_map


def _has_results(actions: pd.DataFrame | None) -> bool:
    """True iff ``actions`` carries a result column the score lookup can read (PA-H1 graceful
    degradation: score_differential is optional match context; without results it stays NaN)."""
    return actions is not None and ("result_id" in actions.columns or "result_name" in actions.columns)


def _in_wide_area(ball_x: float, ball_y: float, goal_x: float, advance_m: float) -> bool:
    if ball_x != ball_x or ball_y != ball_y:  # NaN ball position
        return False
    wide = (ball_y < _WIDE_Y_LOW) or (ball_y > _WIDE_Y_HIGH)
    advanced = abs(ball_x - goal_x) <= advance_m
    return wide and advanced


def prepare_xcross_training_data(
    frames: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    home_team_id,
    feature_set: XCrossFeatureSet = "faithful",
    horizon_seconds: float = 1.0,
    wide_area_only: bool = True,
    advance_m: float = _ADVANCE_M,
    cross_types: tuple[str, ...] = _DEFAULT_CROSS_TYPES,
    carrier_params: dict | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return (features, labels, groups=game_id) for one match. Shared train/serve extractor =
    anti-skew guarantee. ``home_team_id`` is USED to sign score_differential (PA-H1)."""
    # PA-H1: confounder #1 score_differential -- reuse ghost-GK's _build_score_lookup (local import
    # keeps `import _xcross_attempt` light + avoids pulling the ghost-GK model at module load).
    # The callback is _score(game_id, time) -> home_score - away_score (negate for away possessor).
    from silly_kicks.tracking._ghost_gk import _build_score_lookup

    cp: dict = dict(carrier_params) if carrier_params else dict(_DEFAULT_CARRIER_PARAMS)
    carrier = infer_ball_carrier(frames, **cp)
    poss = derive_team_in_possession(frames, carrier)
    goal_map = _build_goal_map(frames)  # H1: precompute once
    score_fn = _build_score_lookup(actions, home_team_id) if _has_results(actions) else None
    has_ball_state = "ball_state" in poss.columns  # M3: column-presence guard (mirror xS:655)

    feat_rows: list[pd.DataFrame] = []
    frame_index_rows: list[dict] = []
    coverage = {"in_domain": 0, "carrier_resolved": 0}  # L-2 carrier-coverage log
    for (gid, pid, fid), grp in poss.groupby(["game_id", "period_id", "frame_id"], sort=False):
        if has_ball_state:
            ball_row = grp[grp["is_ball"]]
            # M3: judge ball_state on the ball row (row-level), not .all() over every player row
            if len(ball_row) and str(ball_row["ball_state"].iloc[0]) == "dead":
                continue
        in_poss = grp["team_in_possession"].dropna()
        if in_poss.empty:
            continue
        poss_team = in_poss.iloc[0]
        # Defending team = the OTHER team's non-ball players. Filter by is_ball + dropna so a real
        # frame whose team_id column carries pd.NA (e.g. the ball row, or an unresolved GS jersey)
        # does not (a) hit "t not in (poss_team, 'ball')" -> "boolean value of NA is ambiguous", or
        # (b) be mistaken for a defending team. Mirrors compute_xcross_attempt's dropna()+!= pattern.
        non_ball = grp[~grp["is_ball"].astype(bool)]
        defending = [t for t in non_ball["team_id"].dropna().unique() if t != poss_team]
        if not defending:
            continue
        goal_x = goal_map.get((gid, pid, defending[0]))
        if goal_x is None:
            continue
        ball = grp[grp["is_ball"]]
        bx = float(ball["x"].iloc[0]) if len(ball) else np.nan
        by = float(ball["y"].iloc[0]) if len(ball) else np.nan
        if wide_area_only and not _in_wide_area(bx, by, goal_x, advance_m):
            continue
        coverage["in_domain"] += 1
        carrier_pid_s = grp["ball_carrier_player_id"].dropna()
        carrier_pid = carrier_pid_s.iloc[0] if not carrier_pid_s.empty else None
        if carrier_pid is not None:
            coverage["carrier_resolved"] += 1
        sd = np.nan  # PA-H1: score differential from the POSSESSING team's perspective
        if score_fn is not None:
            raw = score_fn(gid, float(grp["time_seconds"].iloc[0]))  # home_score - away_score
            # ADR-019: a naive str()==str() renders a float-backed id as "5.0" against a
            # scalar "5", so the compare is ALWAYS False and every row's sign flips.
            sd = raw if same_id(poss_team, home_team_id) else -raw
        feat_rows.append(
            extract_xcross_features(
                grp,
                gk_team_id=defending[0],
                goal_x=goal_x,
                carrier_player_id=carrier_pid,
                feature_set=feature_set,
                score_differential=sd,
            )
        )
        frame_index_rows.append(
            dict(
                game_id=gid,
                period_id=pid,
                frame_id=fid,
                time_seconds=grp["time_seconds"].iloc[0],
                team_in_possession=poss_team,
            )
        )

    if coverage["in_domain"]:
        rate = coverage["carrier_resolved"] / coverage["in_domain"]
        if rate < 0.8:
            warnings.warn(
                f"xCross carrier-resolution coverage {rate:.0%} over the wide-area domain "
                f"({coverage['carrier_resolved']}/{coverage['in_domain']}); GK/confounder "
                f"features may be degraded for this corpus.",
                UserWarning,
                stacklevel=2,
            )

    if not feat_rows:
        empty = pd.DataFrame(columns=XCROSS_FEATURE_NAMES_FAITHFUL)
        return empty, np.array([], dtype=int), np.array([], dtype=object)

    features = pd.concat(feat_rows, ignore_index=True)
    frame_index = pd.DataFrame(frame_index_rows)  # carries team_in_possession (matches label helper)
    y = np.asarray(build_xcross_labels(frame_index, actions, horizon_seconds=horizon_seconds, cross_types=cross_types))
    groups = frame_index["game_id"].to_numpy()
    return features, y, groups


def _chirality_block(model: XCrossAttemptModel) -> dict:
    """Behavioral chirality fingerprint (ADR-037): the model's own extractor + predict on
    the canonical y-asymmetric probe frame. Emitted into save() metadata; a y-mirrored
    artifact cannot reproduce it (the 4.18.0-weights class of bug)."""
    from silly_kicks.tracking._chirality import chirality_fingerprint

    def _predict(frame):
        feats = extract_xcross_features(
            frame, gk_team_id="B", goal_x=105.0, carrier_player_id="A1", score_differential=float("nan")
        )
        return model.predict_proba(feats)

    return chirality_fingerprint(_predict)


def _feature_contract_block() -> dict:
    """Feature contract (ADR-050): this model's FEATURE VECTOR on the fixed probe frame, plus the
    geometry constants its extractor consumes. Mirror of the xS block; see that docstring for why
    it takes no model.

    Declares all three constants it actually reads: the box pair drives ``box_off_def_ratio`` and
    the goal width drives the post distances. The box constants are the canonical
    ``spadlconfig.penalty_area_*`` values as of ADR-050; ``_GOAL_HALF_WIDTH_M`` remains
    module-local (unifying goal width is a separate change), but declaring it is what makes that
    distinction enforceable rather than aspirational.

    Declared from the MODULE ALIASES, not ``_spc.*`` directly: the aliases are what the vectorized
    ``in_box`` predicate actually evaluates, and reading a different binding here would let the two
    drift apart silently.
    """
    from silly_kicks.tracking._feature_contract import contract_probe_frame, feature_contract

    def _vec():
        return (
            extract_xcross_features(
                contract_probe_frame(),
                gk_team_id="B",
                goal_x=105.0,
                carrier_player_id="A2",
                score_differential=1.0,
            )
            .iloc[0]
            .to_numpy(dtype=float)
        )

    return feature_contract(
        _vec,
        constants={
            "penalty_area_half_width": _BOX_HALF_WIDTH_M,
            "penalty_area_depth": _BOX_DEPTH_M,
            "goal_width": _GOAL_HALF_WIDTH_M * 2.0,
        },
    )


def _pinned_params(overrides: dict | None) -> dict:
    """Pinned-deterministic XGBoost params (mirror xS _pinned_params exactly -- M5)."""
    base = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.3,
        "tree_method": "hist",
        "n_jobs": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "random_state": 42,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    if overrides:
        base.update(overrides)
    for k in _INT_PARAMS:
        if base.get(k) is not None:
            base[k] = round(float(base[k]))
    return base


class XCrossAttemptModel:
    """xCrossAttempt classifier: pinned-deterministic XGBoost over snapshot frame features.

    Pickle-free serialization (booster JSON + metadata.json + SHA256SUMS). ``carrier_params`` are
    recorded so inference resolves possession identically to training (R3). See NOTICE.
    """

    def __init__(self, *, feature_set: XCrossFeatureSet = "faithful", params: dict | None = None) -> None:
        if feature_set != "faithful":
            raise NotImplementedError("Only feature_set='faithful' is implemented (TF-17 PR-A).")
        self.feature_set: XCrossFeatureSet = feature_set
        self._params = _pinned_params(params)
        self._booster = None  # xgboost.Booster after fit/load
        self.carrier_params: dict = dict(_DEFAULT_CARRIER_PARAMS)
        self.horizon_seconds: float = 1.0
        self.cross_types: list[str] = list(_DEFAULT_CROSS_TYPES)
        # Provenance (set by the trainer before save(); recorded in metadata).
        self.shipped_variant: str | None = None
        self.provider_list: list | None = None

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        *,
        carrier_params: dict | None = None,
        horizon_seconds: float = 1.0,
    ) -> XCrossAttemptModel:
        """Fit the classifier. Records ``carrier_params`` into the model (R3)."""
        import xgboost as xgb

        if int(xgb.__version__.split(".")[0]) < 2:
            raise RuntimeError("xCrossAttempt requires xgboost>=2.0 (calibrated base_score).")
        self.carrier_params = dict(carrier_params) if carrier_params else dict(_DEFAULT_CARRIER_PARAMS)
        self.horizon_seconds = horizon_seconds
        params = dict(self._params)
        params["base_score"] = float(np.asarray(labels, dtype=float).mean())  # calibrated intercept
        clf = xgb.XGBClassifier(**params)
        clf.fit(features.to_numpy(dtype=float), np.asarray(labels, dtype=int))  # M1: numpy
        booster = clf.get_booster()
        booster.feature_names = list(features.columns)
        self._booster = booster
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return P(cross) per row, shape (n,)."""
        if self._booster is None:
            raise RuntimeError("Model not fitted/loaded.")
        import xgboost as xgb

        dm = xgb.DMatrix(features.to_numpy(dtype=float), feature_names=list(features.columns))
        return np.asarray(self._booster.predict(dm), dtype=float)

    def save(self, path: Path) -> None:
        """Serialize to booster JSON + metadata.json + SHA256SUMS (no pickle)."""
        if self._booster is None:
            raise RuntimeError("Model not fitted.")
        import platform

        import xgboost as xgb

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(path / "model.json"))
        metadata = {
            "feature_names": XCROSS_FEATURE_NAMES_FAITHFUL,
            "feature_set": self.feature_set,
            "horizon_seconds": self.horizon_seconds,
            "cross_types": self.cross_types,
            "carrier_params": self.carrier_params,
            "params": self._params,
            "version": _MODEL_VERSION,
            "pitch_length": _geo.PITCH_LENGTH,
            "pitch_width": _geo.PITCH_WIDTH,
            "geometry_version": _geo.GEOMETRY_VERSION,
            "xgboost_version": xgb.__version__,
            "training_platform": platform.platform(),
            "shipped_variant": self.shipped_variant,
            "provider_list": self.provider_list,
            "chirality": _chirality_block(self),
            "feature_contract": _feature_contract_block(),
        }
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2), newline="\n")
        with open(path / "SHA256SUMS", "w", newline="\n") as f:
            for fname in ["model.json", "metadata.json"]:
                raw = (path / fname).read_bytes()
                if fname.endswith(".json"):
                    raw = raw.replace(b"\r\n", b"\n")
                f.write(f"{hashlib.sha256(raw).hexdigest()}  {fname}\n")

    @classmethod
    def load(cls, path: Path, *, legacy_override: bool = False) -> XCrossAttemptModel:
        """Load from a local directory, verifying SHA-256. Requires xgboost.

        A behavioral chirality fingerprint is enforced (ADR-037 § 9, TF-19 PR-2): a
        pre-PR-2 artifact with no fingerprint is REFUSED unless ``legacy_override=True``
        (which warns), and an output/probe-frame mismatch raises. See ``_chirality``.
        """
        path = Path(path)
        sums = path / "SHA256SUMS"
        if not sums.exists():
            raise IntegrityError(f"SHA256SUMS not found in {path}")
        for line in sums.read_text().splitlines():
            if not line.strip():
                continue
            expected, fname = line.split("  ", 1)
            raw = (path / fname).read_bytes()
            if fname.endswith(".json"):
                raw = raw.replace(b"\r\n", b"\n")
            if hashlib.sha256(raw).hexdigest() != expected:
                raise IntegrityError(f"Integrity check failed for {fname}")
        meta = json.loads((path / "metadata.json").read_text())
        # Coordinate-change guard: pitch-dimension/unit mismatch skews every goal-relative feature ->
        # FAIL CLOSED. A geometry_version change at identical dims is translation-invariant -> warn.
        rec_len = meta.get("pitch_length")
        rec_wid = meta.get("pitch_width")
        if rec_len is not None and (rec_len != _geo.PITCH_LENGTH or rec_wid != _geo.PITCH_WIDTH):
            raise IntegrityError(
                f"Pitch-dimension mismatch: model trained on {rec_len}x{rec_wid} m, library is "
                f"{_geo.PITCH_LENGTH}x{_geo.PITCH_WIDTH} m. Goal-relative features would be skewed; "
                "refusing to load (retrain required)."
            )
        rec_geo = meta.get("geometry_version")
        if rec_geo is not None and rec_geo != _geo.GEOMETRY_VERSION:
            warnings.warn(
                f"geometry_version mismatch (model={rec_geo}, library={_geo.GEOMETRY_VERSION}) at "
                "identical pitch dimensions -- treated as translation-invariant.",
                stacklevel=2,
            )
        model = cls(feature_set=meta.get("feature_set", "faithful"), params=meta.get("params"))
        model.carrier_params = meta.get("carrier_params", dict(_DEFAULT_CARRIER_PARAMS))
        model.horizon_seconds = meta.get("horizon_seconds", 1.0)
        model.cross_types = meta.get("cross_types", model.cross_types)
        model.shipped_variant = meta.get("shipped_variant")
        model.provider_list = meta.get("provider_list")
        model._booster = load_xgb_booster_base_score_safe(path / "model.json")

        from silly_kicks.tracking._chirality import verify_chirality

        verify_chirality(
            _chirality_block(model),
            meta.get("chirality"),
            legacy_override=legacy_override,
            model_name="xCrossAttempt",
        )

        from silly_kicks.tracking._feature_contract import verify_feature_contract

        verify_feature_contract(
            _feature_contract_block(),
            meta.get("feature_contract"),
            legacy_override=legacy_override,
            model_name="xCrossAttempt",
            error_cls=IntegrityError,
        )
        return model

    @classmethod
    def from_variant(cls, variant: str = "default") -> XCrossAttemptModel:
        """Load a bundled variant by name (memoized). ``"public"`` aliases to the bundled
        ``"default"`` (which IS the public arm); only ``"sc_extended"`` falls through to the Hub.
        """
        variant = _VARIANT_ALIASES.get(variant, variant)
        if variant in _VARIANT_CACHE:
            return _VARIANT_CACHE[variant]
        weights_dir = _XCROSS_WEIGHTS_ROOT / variant
        if (weights_dir / "SHA256SUMS").exists():
            model = cls.load(weights_dir)
        elif variant in _HUB_VARIANTS:
            model = cls.from_hub(_HF_REPO_ID)
        else:
            raise FileNotFoundError(
                f"No bundled xCrossAttempt weights for variant {variant!r} at {weights_dir}. "
                "Train via scripts/train_xcross_attempt.py, or await the PR-B weights follow-up."
            )
        _VARIANT_CACHE[variant] = model
        return model

    @classmethod
    def from_hub(cls, repo_id: str = _HF_REPO_ID) -> XCrossAttemptModel:
        """Download published weights from HuggingFace Hub and load.

        Requires ``pip install silly-kicks[xcross]``.

        Examples
        --------
        >>> # model = XCrossAttemptModel.from_hub()
        """
        try:
            from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError("xCrossAttempt Hub weights require: pip install silly-kicks[xcross]") from None
        local_dir = snapshot_download(repo_id=repo_id)
        return cls.load(Path(local_dir))


def _resolve_model(model: XCrossAttemptModel | str | None) -> XCrossAttemptModel:
    if isinstance(model, XCrossAttemptModel):
        return model
    if model is None or isinstance(model, str):
        return XCrossAttemptModel.from_variant(model or "default")  # raises until PR-B weights ship
    raise TypeError(f"Unsupported model type: {type(model)!r}")


def compute_xcross_attempt(
    frames: pd.DataFrame,
    *,
    model: XCrossAttemptModel | str | None = None,
    home_team_id: int | str | None = None,
    actions: pd.DataFrame | None = None,
    pitch_control_cache=None,  # reserved for 'extended' (not used by 'faithful')
    link_frame_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Add an ``xcross_attempt`` column (P(cross within ~horizon)) per in-possession frame.

    STATE-anchored: scores the in-possession team's cross propensity at each frame. Possession +
    defended goal use the model's metadata ``carrier_params`` (R3). ``actions`` (optional, PA-H1)
    enables ``score_differential`` via the match-context score lookup; omitted -> NaN
    (XGBoost-tolerant). ``pitch_control_cache`` is reserved for the deferred 'extended' variant.
    """
    m = _resolve_model(model)
    out = frames.copy()
    out["xcross_attempt"] = np.nan

    # N-A (mirror xS): carrier inference + possession run on the FULL contiguous frames (cross-frame
    # hysteresis); restrict ONLY the per-frame extract + batched predict to link_frame_ids.
    carrier = infer_ball_carrier(frames, **m.carrier_params)
    poss = derive_team_in_possession(frames, carrier)
    goal_map = _build_goal_map(frames)  # H1: once
    score_fn = None
    if actions is not None and _has_results(actions) and home_team_id is not None:
        from silly_kicks.tracking._ghost_gk import _build_score_lookup

        score_fn = _build_score_lookup(actions, home_team_id)

    feat_rows: list[pd.DataFrame] = []
    keys: list[tuple] = []  # (gid, pid, frame_id, team_in_possession)
    for (gid, pid, frame_id), grp in poss.groupby(["game_id", "period_id", "frame_id"], dropna=False):
        if link_frame_ids is not None and int(str(frame_id)) not in link_frame_ids:
            continue
        tip = grp["team_in_possession"].iloc[0]
        if pd.isna(tip):
            continue
        outfield = grp[~grp["is_ball"].astype(bool)]
        teams = [t for t in outfield["team_id"].dropna().unique() if t != tip]
        if not teams:
            continue
        def_team = teams[0]
        goal_x = goal_map.get((gid, pid, def_team))
        if goal_x is None:
            continue
        carrier_pid_s = grp["ball_carrier_player_id"].dropna()
        carrier_pid = carrier_pid_s.iloc[0] if not carrier_pid_s.empty else None
        sd = np.nan
        if score_fn is not None:
            raw = score_fn(gid, float(grp["time_seconds"].iloc[0]))
            # ADR-019: see the sibling site above -- a naive str compare flips every sign.
            sd = raw if same_id(tip, home_team_id) else -raw
        feat_rows.append(
            extract_xcross_features(
                grp, gk_team_id=def_team, goal_x=goal_x, carrier_player_id=carrier_pid, score_differential=sd
            )
        )
        keys.append((gid, pid, frame_id, tip))

    if not feat_rows:
        return out

    feature_matrix = pd.concat(feat_rows, ignore_index=True)
    probs = m.predict_proba(feature_matrix)
    key_df = pd.DataFrame(keys, columns=["game_id", "period_id", "frame_id", "team_id"])
    key_df["__p"] = probs
    # N-B (mirror xS): join on TEMPORARY canonical id keys -> never mutate out dtypes.
    # canonical_id_series (ADR-019) is the dtype-safe stringify (366/366.0/Int64(366) -> "366"),
    # avoiding the .astype(str) "366.0" artifact on any float-upcast id column.
    out["__gid"] = canonical_id_series(out["game_id"])
    out["__tid"] = canonical_id_series(out["team_id"])
    key_df["__gid"] = canonical_id_series(key_df["game_id"])
    key_df["__tid"] = canonical_id_series(key_df["team_id"])
    key_df = key_df.drop(columns=["game_id", "team_id"])
    out = out.merge(key_df, on=["__gid", "period_id", "frame_id", "__tid"], how="left")
    out["xcross_attempt"] = out["__p"]
    return out.drop(columns=["__p", "__gid", "__tid"])


@nan_safe_enrichment
def add_xcross_attempt(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    model: XCrossAttemptModel | str | None = None,
    links: pd.DataFrame | None = None,
    home_team_id: int | str | None = None,
    actions_for_context: pd.DataFrame | None = None,
    pitch_control_cache=None,
) -> pd.DataFrame:
    """Enrich SPADL actions with an ``xcross_attempt`` column (xCross at the linked frame).

    xCross is the **possessing team's** cross propensity; a non-possessing-team action at the same
    frame gets NaN by design. NaN identifiers route to NaN output (ADR-003). ``links`` skips internal
    linking. ``actions_for_context`` (optional, PA-H1) threads match-context to score_differential
    at compute time; defaults to ``actions`` when scoring fresh.
    """
    m = _resolve_model(model)
    out = actions.copy()
    pointers = links if links is not None else link_actions_to_frames(actions, frames)[0]

    link_frame_ids = None
    if "frame_id" in pointers.columns:
        link_frame_ids = {int(f) for f in pointers["frame_id"].dropna().astype(int).tolist()}

    if "xcross_attempt" in frames.columns and frames["xcross_attempt"].notna().any():
        scored = frames
    else:
        scored = compute_xcross_attempt(
            frames,
            model=m,
            home_team_id=home_team_id,
            actions=actions_for_context if actions_for_context is not None else actions,
            link_frame_ids=link_frame_ids,
        )

    xcol = scored[scored["xcross_attempt"].notna()][
        ["game_id", "period_id", "frame_id", "team_id", "xcross_attempt"]
    ].copy()
    linked = pointers.merge(actions[["action_id", "game_id", "period_id", "team_id"]], on="action_id", how="left")
    # ADR-019: action-side ids (e.g. int64 team_id) vs frame-derived ids (Int64/object) on the
    # merge keys. align_join_keys no-ops when both sides are merge-compatible (both numeric or both
    # object) and canonicalizes only the numeric-vs-object keys -- fixing the silent cross-dtype miss.
    merge_keys = ["game_id", "period_id", "frame_id", "team_id"]
    linked, xcol = align_join_keys(linked, xcol, merge_keys)
    merged = linked.merge(xcol, on=merge_keys, how="left")
    deduped = merged.drop_duplicates(subset=["action_id"], keep="first")
    col = deduped.set_index("action_id")["xcross_attempt"]
    out = out.merge(col.rename("xcross_attempt"), left_on="action_id", right_index=True, how="left")
    return out


def xcross_attempt_xfns(
    *,
    model: XCrossAttemptModel | str | None = None,
    home_team_id: int | str | None = None,
    pitch_control_cache=None,
) -> list:
    """Factory returning a FrameAwareTransformer emitting xcross_attempt_a0/_a1/_a2.

    NOT added to any default/union xfn list until PR-B weights ship.
    """
    cols = ["xcross_attempt_a0", "xcross_attempt_a1", "xcross_attempt_a2"]

    def _transformer(states, frames):
        out = pd.DataFrame(index=states[0].index)
        if frames is None:
            for c in cols:
                out[c] = np.nan
            return out
        m = _resolve_model(model)
        slot_pointers = []
        link_frame_ids: set[int] = set()
        for slot in states[:3]:
            ptr = link_actions_to_frames(slot, frames)[0]
            slot_pointers.append(ptr)
            if "frame_id" in ptr.columns:
                link_frame_ids |= {int(f) for f in ptr["frame_id"].dropna().astype(int).tolist()}
        scored = compute_xcross_attempt(frames, model=m, home_team_id=home_team_id, link_frame_ids=link_frame_ids)
        for i, (slot, ptr) in enumerate(zip(states[:3], slot_pointers, strict=False)):
            enriched = add_xcross_attempt(slot, scored, model=m, home_team_id=home_team_id, links=ptr)
            out[cols[i]] = enriched["xcross_attempt"].to_numpy() if "xcross_attempt" in enriched else np.nan
        return out

    _transformer._frame_aware = True  # type: ignore[attr-defined]
    _transformer.__name__ = "xcross_attempt_xfn"
    return [_transformer]
