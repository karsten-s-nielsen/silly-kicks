"""Measured positioning-gap metric (spec section 6).

``positioning_gap = threat(actual shape) - threat(reachable optimum) >= 0`` per defensive frame:
how much attacking threat the realized shape failed to suppress versus the best REACHABLE
repositioning. Frame/team grain (the defending unit's positioning efficiency).

Hexagonal: imports ``silly_kicks.tracking`` PUBLIC seams + ``silly_kicks.id_compat`` +
``silly_kicks.spadl.config`` (pitch geometry, the gkdv precedent) only. ``xt`` is INJECTED.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Collection

import numpy as np
import pandas as pd

from silly_kicks.id_compat import canonical_id, same_id
from silly_kicks.spadl import config as spadlconfig

# Public tracking seams.
from silly_kicks.tracking import (
    SPEED_SOURCE_UNAVAILABLE,
    GoalEndUnresolvedError,
    infer_ball_carrier,
    resolve_defended_goals,
)

from ._config import PositioningParams
from ._constraints import ReachabilityConstraint
from ._objectives import Objective, ThreatObjective
from ._report import POSITIONING_GAP_SOURCE_VALUES, PositioningReport
from ._solve import optimise_positions

_FRAME_KEYS = ["game_id", "period_id", "frame_id"]
_GOAL_Y = spadlconfig.field_width / 2.0  # 34.0 -- the goal's y-centre

#: Samples columns in emission order (the family's full declared output).
POSITIONING_SAMPLE_COLUMNS = [
    "game_id",
    "period_id",
    "frame_id",
    "team_id",
    "positioning_gap",
    "threat_actual",
    "threat_optimum",
    "n_movable",
    "n_feasible_proposals",
    "sa_converged",
    "positioning_gap_source",
]

#: Mart grain (the ``summarize_positioning_gap`` grain -- the defending unit per match; ADR-098).
POSITIONING_KEYS: tuple[str, ...] = ("game_id", "team_id")
#: The glossaried value columns (ADR-098; provenance/count columns are not metric columns).
POSITIONING_METRIC_COLUMNS: tuple[str, ...] = ("positioning_gap", "threat_actual", "threat_optimum")


def _seed(game_id, period_id, frame_id) -> int:
    """A STABLE (process-independent) RNG seed from the frame key, so the column is CI-golden-able.

    Python's ``hash`` is salted per process; ``hashlib`` is not.
    """
    token = f"{canonical_id(game_id)}|{canonical_id(period_id)}|{canonical_id(frame_id)}"
    return int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)


def _downsample(frames: pd.DataFrame, sample_fps: float) -> pd.DataFrame:
    """Keep at most one frame per ``(game_id, period_id, floor(time_seconds * sample_fps))`` bucket.

    25 fps tracking is massively autocorrelated; 1 fps (the default) is the honest sampling grain.
    Non-positive ``sample_fps`` disables down-sampling.
    """
    if sample_fps is None or sample_fps <= 0 or "time_seconds" not in frames.columns:
        return frames
    keyframes = frames[[*_FRAME_KEYS, "time_seconds"]].drop_duplicates(subset=_FRAME_KEYS)
    keyframes = keyframes.assign(_bucket=np.floor(keyframes["time_seconds"].astype(float) * float(sample_fps)))
    kept = keyframes.sort_values([*_FRAME_KEYS]).drop_duplicates(
        subset=["game_id", "period_id", "_bucket"], keep="first"
    )
    kept_keys = {
        (canonical_id(g), canonical_id(p), canonical_id(f))
        for g, p, f in zip(kept["game_id"], kept["period_id"], kept["frame_id"], strict=True)
    }
    mask = [
        (canonical_id(g), canonical_id(p), canonical_id(f)) in kept_keys
        for g, p, f in zip(frames["game_id"], frames["period_id"], frames["frame_id"], strict=True)
    ]
    return frames[pd.Series(mask, index=frames.index)]


def _possession_by_key(carrier: pd.DataFrame) -> dict:
    """Canonical ``(game, period, frame) -> ball_carrier_team_id`` (raw id kept as the value)."""
    out: dict = {}
    for g, p, f, team in zip(
        carrier["game_id"], carrier["period_id"], carrier["frame_id"], carrier["ball_carrier_team_id"], strict=True
    ):
        out[(canonical_id(g), canonical_id(p), canonical_id(f))] = None if pd.isna(team) else team
    return out


def _distinct_teams(nonball: pd.DataFrame) -> list:
    """Distinct RAW team ids among non-ball rows (de-duplicated on the canonical id)."""
    seen: set = set()
    teams: list = []
    for t in nonball["team_id"].dropna():
        c = canonical_id(t)
        if c not in seen:
            seen.add(c)
            teams.append(t)
    return teams


def _drop_row(key, source: str, defending, *, n_movable: object = pd.NA) -> dict:
    g, p, f = key
    return {
        "game_id": g,
        "period_id": p,
        "frame_id": f,
        "team_id": defending if defending is not None else pd.NA,
        "positioning_gap": np.nan,
        "threat_actual": np.nan,
        "threat_optimum": np.nan,
        "n_movable": n_movable,
        "n_feasible_proposals": pd.NA,
        "sa_converged": pd.NA,
        "positioning_gap_source": source,
    }


def compute_positioning_gap(
    frames: pd.DataFrame,
    *,
    xt,
    movable: Collection | None = None,
    objective: Objective | None = None,
    params: PositioningParams = PositioningParams.default(),  # noqa: B008 - frozen singleton default
) -> tuple[pd.DataFrame, PositioningReport]:
    """Per defensive frame, ``positioning_gap = threat(actual) - threat(reachable optimum) >= 0``.

    Domain (spec section 6): alive ball, opponent-in-possession, ball within
    ``params.domain_ball_to_goal_m`` of the DEFENDED goal, two-team frame, velocity present, and at
    least one movable defender. Out-of-domain frames are excluded-and-COUNTED with a per-condition
    ``positioning_gap_source`` token; ``PositioningReport`` conserves
    (``n_frames_scored + sum(drop_reasons) == n_frames_in``, ADR-042). Honest-NaN -- never a
    fabricated 0.

    ``xt`` is the INJECTED fitted ExpectedThreat (the port pattern). ``movable`` defaults to the
    defending team's OUTFIELDERS (the keeper stays a fixed agent, GKDV's domain). ``objective``
    defaults to a per-frame :class:`ThreatObjective` oriented via ``resolve_defended_goals`` -- the
    shipped-column objective; a supplied ``objective`` is used verbatim for every frame (advanced /
    exploratory, e.g. a carrier-locating :class:`PressureObjective`). The RNG is seeded from
    ``(game_id, period_id, frame_id)`` so the column is a pure function of inputs.

    Velocity (ADR-063): a declared-velocity-unavailable frame (``speed_source`` marker) is excluded
    as ``velocity_unavailable``; frames with ``vx``/``vy`` columns entirely absent and NOT declared
    unavailable RAISE (the forgot-``derive_velocities()`` caller bug, fail-loud).

    Returns ``(samples, report)`` where ``samples`` has one row per evaluated (down-sampled) frame.

    Examples
    --------
    Score the reachable positioning gap for a defending unit::

        from silly_kicks.positioning import compute_positioning_gap

        samples, report = compute_positioning_gap(frames, xt=fitted_xt)
        scored = samples[samples["positioning_gap_source"] == "scored"]
        assert report.conserves()
    """
    has_vxvy = "vx" in frames.columns and "vy" in frames.columns
    declared_all = (
        "speed_source" in frames.columns
        and len(frames) > 0
        and bool((frames["speed_source"] == SPEED_SOURCE_UNAVAILABLE).all())
    )
    if not has_vxvy and not declared_all:
        raise ValueError(
            "compute_positioning_gap: frames are missing velocity columns 'vx'/'vy' and are not "
            "declared velocity-unavailable (speed_source). Run "
            "silly_kicks.tracking.preprocess.derive_velocities(frames) first (ADR-063)."
        )

    work = _downsample(frames, params.sample_fps)
    if len(work) == 0:
        empty = pd.DataFrame(columns=POSITIONING_SAMPLE_COLUMNS)
        return empty, PositioningReport(params=params, n_frames_in=0, n_frames_scored=0, drop_reasons={})

    goal_map = resolve_defended_goals(work)  # ONCE (ADR-055), threaded everywhere
    possession = _possession_by_key(infer_ball_carrier(work))

    rows: list[dict] = []
    for key, grp in work.groupby(_FRAME_KEYS, sort=True):
        rows.append(
            _classify_and_score(
                grp,
                key,
                goal_map=goal_map,
                possession=possession,
                params=params,
                xt=xt,
                movable=movable,
                objective=objective,
            )
        )

    samples = _finalize_samples(rows)
    n_in = len(samples)
    n_scored = int((samples["positioning_gap_source"] == "scored").sum())
    drop_reasons = {
        str(k): int(v)
        for k, v in Counter(
            samples.loc[samples["positioning_gap_source"] != "scored", "positioning_gap_source"]
        ).items()
    }
    report = PositioningReport(params=params, n_frames_in=n_in, n_frames_scored=n_scored, drop_reasons=drop_reasons)
    return samples, report


def _classify_and_score(grp, key, *, goal_map, possession, params, xt, movable, objective) -> dict:
    g, p, f = key
    nonball = grp[~grp["is_ball"].astype(bool)]

    # 1) two-team frame (structural precondition, distinct from out-of-domain)
    teams = _distinct_teams(nonball)
    if len(teams) != 2:
        return _drop_row(key, "excluded_not_two_teams", None)

    # 2) velocity declared-unavailable (SB360 shape) -> excluded-and-counted (ADR-063)
    if "speed_source" in grp.columns and len(grp) > 0 and bool((grp["speed_source"] == SPEED_SOURCE_UNAVAILABLE).all()):
        return _drop_row(key, "velocity_unavailable", None)

    # 3) possession (opponent) + defending team
    poss = possession.get((canonical_id(g), canonical_id(p), canonical_id(f)))
    if poss is None or not any(same_id(t, poss) for t in teams):
        return _drop_row(key, "excluded_out_of_domain", None)
    defending_list = [t for t in teams if not same_id(t, poss)]
    if len(defending_list) != 1:
        return _drop_row(key, "excluded_not_two_teams", None)
    defending = defending_list[0]

    # 4) goal geometry (the DEFENDED goal); unresolved -> honest-NaN (ADR-055)
    try:
        defended_x = goal_map.get(g, p, defending, allow_guess=True)
    except GoalEndUnresolvedError:
        defended_x = None
    if defended_x is None:
        return _drop_row(key, "unresolved_geometry", defending)

    # 5) alive ball within the danger domain of the defended goal
    ball = grp[grp["is_ball"].astype(bool)]
    if len(ball) == 0:
        return _drop_row(key, "excluded_out_of_domain", defending)
    br = ball.iloc[0]
    ball_state = br.get("ball_state") if "ball_state" in grp.columns else None
    if ball_state is not None and pd.notna(ball_state) and str(ball_state) != "alive":
        return _drop_row(key, "excluded_out_of_domain", defending)
    bx, by = float(br["x"]), float(br["y"])
    dist = float(np.hypot(bx - float(defended_x), by - _GOAL_Y))
    if not np.isfinite(dist) or dist > float(params.domain_ball_to_goal_m):
        return _drop_row(key, "excluded_out_of_domain", defending)

    # 6) movable defenders (default: the defending team's outfielders; GK stays a fixed agent)
    if movable is None:
        mov = list(
            nonball.loc[
                (~nonball["is_goalkeeper"].astype(bool)) & nonball["team_id"].map(lambda t: same_id(t, defending)),
                "player_id",
            ].dropna()
        )
    else:
        mov = list(movable)
    if len(mov) == 0:
        return _drop_row(key, "degenerate_no_movable", defending, n_movable=0)

    # 7) score: one optimise call yields both actual (incumbent-0) and best (path identity)
    obj = objective if objective is not None else ThreatObjective(xt=xt, goal_map=goal_map, attacking_team_id=poss)
    try:
        res = optimise_positions(
            grp,
            movable=mov,
            objective=obj,
            constraints=[ReachabilityConstraint(params.reachability)],
            seed=_seed(g, p, f),
        )
    except GoalEndUnresolvedError:
        return _drop_row(key, "unresolved_geometry", defending)
    if res.n_feasible_proposals == 0:  # every movable pinned -> no reachable move
        return _drop_row(key, "degenerate_no_movable", defending, n_movable=len(mov))

    return {
        "game_id": g,
        "period_id": p,
        "frame_id": f,
        "team_id": defending,
        "positioning_gap": float(res.actual_score - res.best_score),
        "threat_actual": float(res.actual_score),
        "threat_optimum": float(res.best_score),
        "n_movable": len(mov),
        "n_feasible_proposals": int(res.n_feasible_proposals),
        "sa_converged": bool(res.converged),
        "positioning_gap_source": "scored",
    }


def _summary_drop_columns() -> list[str]:
    return [f"n_{r}" for r in POSITIONING_GAP_SOURCE_VALUES if r != "scored"]


def summarize_positioning_gap(samples: pd.DataFrame) -> pd.DataFrame:
    """Aggregate :func:`compute_positioning_gap` samples per ``(game_id, team_id)``.

    ``mean_positioning_gap`` is the mean over SCORED rows only (never over NaN-gap drops);
    ``n_scored`` + one ``n_<reason>`` count per drop reason. Grouped on the CANONICAL id, raw id
    emitted (ADR-019). The defending team of drop rows that could not resolve a team is NA.

    Examples
    --------
    Roll the per-frame samples up to the defending unit per match::

        samples, _ = compute_positioning_gap(frames, xt=fitted_xt)
        summary = summarize_positioning_gap(samples)  # one row per (game_id, team_id)
        worst = summary.sort_values("mean_positioning_gap", ascending=False)
    """
    cols = ["game_id", "team_id", "mean_positioning_gap", "n_scored", *_summary_drop_columns()]
    if len(samples) == 0:
        return pd.DataFrame(columns=cols)

    df = samples.copy()
    df["_g"] = df["game_id"].map(canonical_id)
    df["_t"] = df["team_id"].map(lambda t: pd.NA if pd.isna(t) else canonical_id(t))

    out_rows: list[dict] = []
    for _, grp in df.groupby(["_g", "_t"], dropna=False, sort=True):
        scored = grp[grp["positioning_gap_source"] == "scored"]
        row = {
            "game_id": grp["game_id"].iloc[0],
            "team_id": grp["team_id"].iloc[0],
            "mean_positioning_gap": float(scored["positioning_gap"].mean()) if len(scored) else np.nan,
            "n_scored": len(scored),
        }
        for reason in POSITIONING_GAP_SOURCE_VALUES:
            if reason == "scored":
                continue
            row[f"n_{reason}"] = int((grp["positioning_gap_source"] == reason).sum())
        out_rows.append(row)
    return pd.DataFrame(out_rows, columns=cols)


def _finalize_samples(rows: list[dict]) -> pd.DataFrame:
    samples = pd.DataFrame(rows, columns=POSITIONING_SAMPLE_COLUMNS)
    samples["positioning_gap"] = samples["positioning_gap"].astype("float64")
    samples["threat_actual"] = samples["threat_actual"].astype("float64")
    samples["threat_optimum"] = samples["threat_optimum"].astype("float64")
    samples["n_movable"] = samples["n_movable"].astype("Int64")
    samples["n_feasible_proposals"] = samples["n_feasible_proposals"].astype("Int64")
    samples["sa_converged"] = samples["sa_converged"].astype("boolean")
    samples["positioning_gap_source"] = samples["positioning_gap_source"].astype("object")
    # Every token is drawn from the closed set by construction (_drop_row + "scored"); the guard is
    # test_source_tokens_are_closed_and_per_condition, so a bare assert here would be redundant.
    return samples
