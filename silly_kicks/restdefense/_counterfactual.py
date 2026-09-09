"""restdefense Layer-3 counterfactual engine (TF-60, ADR-089).

A gkdv **sibling**, not a gkdv call: gkdv's ``build_ghost_frames`` ghosts the DEFENDING keeper
when the ball is near the ATTACKED goal; rest defense ghosts the IN-POSSESSION team A's OWN
rearguard (``which="rearguard"``) or keeper (``which="keeper"``) near A's OWN goal, gated on A
being committed-forward. Only the model serves are reused (they already return frame-ready
``ghost_x``/``ghost_y`` under the ADR-089 both-axes inverse), so this engine never re-derives
orientation. PURE -- ``frames`` is never mutated.

Depends on ``silly_kicks.tracking`` PUBLIC seams + ``silly_kicks.id_compat`` +
``silly_kicks._frame_index`` ONLY (ADR-037). See NOTICE for citations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from silly_kicks.tracking import GoalMap

from silly_kicks.id_compat import align_join_keys, canonical_id, ids_equal
from silly_kicks.spadl import config as spadlconfig

from ._columns import RD_FRAME_KEYS
from ._config import RestDefenseParams
from ._ghost_report import RestDefenseGhostReport

_FIELD_LENGTH = float(spadlconfig.field_length)  # 105.0
_GOAL_Y = float(spadlconfig.field_width) / 2.0  # 34.0

# Drop reasons -- a frame the counterfactual cannot simulate is dropped-AND-COUNTED, never scored
# as Delta = 0 (a zero delta from a vanished rearguard reads as "no deterrence", biasing aggregates).
_DROP_BALL_MISSING = "ball_row_missing"
_DROP_BALL_NOT_ALIVE = "ball_not_alive"
_DROP_BALL_COORDS = "ball_coordinates_missing"
_DROP_NO_POSSESSION = "no_possession"
_DROP_NO_GOAL_MAP = "no_goal_map_entry"
_DROP_NOT_COMMITTED = "not_committed_forward"
_DROP_INSUFFICIENT = "insufficient_rearguard"
_DROP_NO_GHOST = "no_ghost_served"

_PROVENANCE_COLUMNS = [
    "game_id",
    "period_id",
    "frame_id",
    "team_id",
    "player_id",
    "ghost_gr_x",
    "ghost_gr_y",
    "ghost_x",
    "ghost_y",
    "actual_x",
    "actual_y",
    "displacement_m",
    "ghost_source",
    "drop_reason",
]

_DEFAULT_PARAMS = RestDefenseParams()


def _pin_carrier(frames: pd.DataFrame, carrier: pd.DataFrame | None):
    from silly_kicks.tracking import infer_ball_carrier

    if carrier is not None:
        return carrier[[*RD_FRAME_KEYS, "ball_carrier_team_id"]].drop_duplicates(subset=RD_FRAME_KEYS, keep="first")
    inferred = infer_ball_carrier(frames)
    return inferred[[*RD_FRAME_KEYS, "ball_carrier_team_id"]].drop_duplicates(subset=RD_FRAME_KEYS, keep="first")


def _apply_domain(
    frames: pd.DataFrame,
    *,
    carrier: pd.DataFrame,
    goal_map: GoalMap,
    params: RestDefenseParams,
    which: Literal["keeper", "rearguard"],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split frame keys into ``(eligible, dropped)`` per the spec domain, one row per frame on both
    sides so they partition the input exactly (the conservation the report rests on)."""
    players = frames[~frames["is_ball"].astype(bool)]
    ball = frames[frames["is_ball"].astype(bool)]

    keys = frames[RD_FRAME_KEYS].drop_duplicates().sort_values(RD_FRAME_KEYS).reset_index(drop=True)

    ball_slim = ball.drop_duplicates(subset=RD_FRAME_KEYS, keep="first")[[*RD_FRAME_KEYS, "x", "ball_state"]].rename(
        columns={"x": "ball_x"}
    )
    work = keys.merge(ball_slim, on=RD_FRAME_KEYS, how="left")

    left, right = align_join_keys(work, carrier, list(RD_FRAME_KEYS))
    work = left.merge(right, on=RD_FRAME_KEYS, how="left").rename(
        columns={"ball_carrier_team_id": "possession_team_id"}
    )

    # A's own defended goal (the end A keeps its rearguard in front of).
    work["own_goal_x"] = [
        _goal_lookup(goal_map, g, p, t)
        for g, p, t in zip(work["game_id"], work["period_id"], work["possession_team_id"], strict=True)
    ]

    # rearguard / keeper presence per frame for the in-possession team A (finite coordinates).
    present = _count_present(players, work, which=which, n_rearguard=params.n_rearguard)

    reason = pd.Series(pd.NA, index=work.index, dtype="object")

    def _mark(mask: np.ndarray, why: str) -> None:
        reason.loc[reason.isna() & pd.Series(mask, index=work.index)] = why

    advance = (work["ball_x"] - work["own_goal_x"]).abs()
    _mark(work["ball_state"].isna().to_numpy(), _DROP_BALL_MISSING)
    _mark((work["ball_state"] == "dead").to_numpy(), _DROP_BALL_NOT_ALIVE)
    _mark(work["ball_x"].isna().to_numpy(), _DROP_BALL_COORDS)
    _mark(work["possession_team_id"].isna().to_numpy(), _DROP_NO_POSSESSION)
    _mark(work["own_goal_x"].isna().to_numpy(), _DROP_NO_GOAL_MAP)
    _mark(~(advance >= float(params.min_ball_advance_m)).fillna(False).to_numpy(), _DROP_NOT_COMMITTED)
    _mark(~present.to_numpy(dtype=bool), _DROP_INSUFFICIENT if which == "rearguard" else "no_keeper")

    work["drop_reason"] = reason
    eligible = work[work["drop_reason"].isna()].drop(columns=["drop_reason"]).reset_index(drop=True)
    dropped = work[work["drop_reason"].notna()][[*RD_FRAME_KEYS, "drop_reason"]].reset_index(drop=True)
    return eligible, dropped


def _count_present(players: pd.DataFrame, work: pd.DataFrame, *, which: str, n_rearguard: int) -> pd.Series:
    """Per-frame boolean: does A have >= n_rearguard finite field defenders (rearguard) / a keeper?"""
    finite = players[players["x"].notna() & players["y"].notna()]
    if which == "keeper":
        pool = finite[finite["is_goalkeeper"].astype(bool)]
    else:
        pool = finite[~finite["is_goalkeeper"].astype(bool)]
    # count A's players per frame, then compare to the per-frame possession team.
    counts = pool.groupby([*RD_FRAME_KEYS, "team_id"], dropna=False).size().rename("n").reset_index()
    left, right = align_join_keys(
        work[[*RD_FRAME_KEYS, "possession_team_id"]].rename(columns={"possession_team_id": "team_id"}),
        counts,
        [*RD_FRAME_KEYS, "team_id"],
    )
    merged = left.merge(right, on=[*RD_FRAME_KEYS, "team_id"], how="left")
    n = merged["n"].fillna(0).to_numpy(dtype=int)
    need = 1 if which == "keeper" else int(n_rearguard)
    return pd.Series(n >= need, index=work.index)


def _goal_lookup(goal_map: GoalMap, game_id, period_id, team_id) -> float:
    if team_id is None or (isinstance(team_id, float) and np.isnan(team_id)) or pd.isna(team_id):
        return float("nan")
    end = goal_map.get(game_id, period_id, team_id, allow_guess=True)
    return float("nan") if end is None else float(end)


def build_restdefense_ghost_frames(
    frames: pd.DataFrame,
    *,
    which: Literal["keeper", "rearguard"],
    model,
    home_team_id: int | str,
    carrier: pd.DataFrame | None = None,
    visible_area: pd.DataFrame | None = None,
    params: RestDefenseParams = _DEFAULT_PARAMS,
) -> tuple[pd.DataFrame, pd.DataFrame, RestDefenseGhostReport]:
    """Ghost the in-possession team A's OWN rearguard (or keeper) to a league-average model.

    PURE: never mutates ``frames``. Returns ``(counterfactual_frames, provenance, report)``.
    ``counterfactual_frames`` is the FULL input with only A's substituted rows moved; consumers MUST
    restrict to the scored set (``provenance["drop_reason"].isna()``) before differencing. A NaN /
    missing / ``variant_unavailable`` / ``fov_cropped`` ghost on an in-domain frame is
    **dropped-and-counted**, never scored as Delta = 0; a non-finite served ghost on a scored frame
    **raises** (pitch control silently drops NaN rows, so a NaN ghost would make the player vanish).

    ``which="rearguard"`` consumes :func:`serve_ghost_outfield_positions`; ``which="keeper"`` consumes
    :func:`serve_ghost_gk_positions` restricted to A's keeper (the model should be a ``sweeper`` variant).

    Examples
    --------
    Ghost the in-possession team's rearguard on LTR-normalised tracking ``frames`` and restrict to the
    scored set before differencing (conservation holds exactly)::

        cf, provenance, report = build_restdefense_ghost_frames(
            frames, which="rearguard", model=ghost_outfield_model, home_team_id=1
        )
        scored = provenance[provenance["drop_reason"].isna()]
        assert report.n_frames_scored + sum(report.drop_reasons.values()) == report.n_frames_in
    """
    from silly_kicks.tracking import (
        resolve_defended_goals,
        serve_ghost_gk_positions,
        serve_ghost_outfield_positions,
    )

    if which not in ("keeper", "rearguard"):
        raise ValueError(f"which must be 'keeper' or 'rearguard', got {which!r}")

    src = frames  # never mutated
    car = _pin_carrier(src, carrier)
    goal_map = resolve_defended_goals(src)
    eligible, dropped = _apply_domain(src, carrier=car, goal_map=goal_map, params=params, which=which)

    if which == "rearguard":
        served = serve_ghost_outfield_positions(
            src,
            model=model,
            home_team_id=home_team_id,
            carrier=car,
            n_rearguard=params.n_rearguard,
            visible_area=visible_area,
        )
        served = served.rename(columns={"ghost_outfield_source": "ghost_source"})
    else:
        served = serve_ghost_gk_positions(src, model=model, home_team_id=home_team_id, carrier=car)
        served = served.rename(columns={"gk_team_id": "team_id"})
        served["ghost_source"] = "computed"
        # The GK serve is keyed (frame, team) with NO player_id, but the provenance/write-back match
        # on (frame, team, player_id) -- attach A's actual keeper player_id per (frame, team) so the
        # keeper is substituted (without this the served row joins nothing and the eligible frame falls
        # to `no_ghost_served`). A substitution window (two keepers in one frame) is deduped to one,
        # keep="first", mirroring the gkdv dose-target dedup.
        non_ball = src[~src["is_ball"].astype(bool)]
        gk_rows = non_ball[non_ball["is_goalkeeper"].astype(bool)].drop_duplicates(
            subset=[*RD_FRAME_KEYS, "team_id"], keep="first"
        )[[*RD_FRAME_KEYS, "team_id", "player_id"]]
        left, right = align_join_keys(served, gk_rows, [*RD_FRAME_KEYS, "team_id"])
        served = left.merge(right, on=[*RD_FRAME_KEYS, "team_id"], how="left")
        # restrict to A's (in-possession) keeper -- the serve returns BOTH teams'.
        pk = eligible[[*RD_FRAME_KEYS, "possession_team_id"]]
        left, right = align_join_keys(served, pk, list(RD_FRAME_KEYS))
        m = left.merge(right, on=RD_FRAME_KEYS, how="inner")
        # column-vs-column (ids_equal), NOT ids_match (Series-vs-SCALAR: passing a Series as the scalar
        # compares every row against the whole Series object -> always False -> the keeper is never
        # restricted and the frame falls to `no_ghost_served`).
        keep = ids_equal(m["team_id"], m["possession_team_id"]).fillna(False).to_numpy(dtype=bool)
        served = m[keep].drop(columns=["possession_team_id"])

    _raise_on_bug_ghost(served, eligible)
    prov = _build_provenance(src, served=served, eligible=eligible, dropped=dropped)
    cf = _write_back(src, provenance=prov)

    scored_rows = prov[prov["drop_reason"].isna()]
    n_scored_frames = len(scored_rows[RD_FRAME_KEYS].drop_duplicates())
    drop_reasons = {str(k): int(v) for k, v in prov["drop_reason"].dropna().value_counts().to_dict().items()}
    report = RestDefenseGhostReport(
        params=params,
        n_frames_in=len(src[RD_FRAME_KEYS].drop_duplicates()),
        n_frames_scored=n_scored_frames,
        drop_reasons=drop_reasons,
    )
    return cf, prov, report


def _raise_on_bug_ghost(served: pd.DataFrame, eligible: pd.DataFrame) -> None:
    """Raise if a ``computed`` ghost is non-finite on an in-domain frame -- a serve BUG, distinct from
    an honest ``variant_unavailable``/``fov_cropped`` NaN (which is dropped-and-counted). Pitch control
    silently drops NaN-coordinate rows, so a computed-NaN ghost would make the player vanish rather
    than error (mirrors the gkdv guard)."""
    if not len(served) or "ghost_source" not in served.columns:
        return
    left, right = align_join_keys(served, eligible[RD_FRAME_KEYS], list(RD_FRAME_KEYS))
    es = left.merge(right, on=RD_FRAME_KEYS, how="inner")
    if not len(es):
        return
    computed = es[es["ghost_source"].astype(str) == "computed"]
    if len(computed) and not np.isfinite(computed[["ghost_x", "ghost_y"]].to_numpy(dtype=float)).all():
        raise ValueError(
            "build_restdefense_ghost_frames produced a non-finite 'computed' ghost coordinate on an "
            "in-domain frame. Pitch control silently DROPS NaN-coordinate rows, so a NaN ghost would "
            "make the player vanish rather than error (mirrors the gkdv guard)."
        )


def _build_provenance(
    frames: pd.DataFrame,
    *,
    served: pd.DataFrame,
    eligible: pd.DataFrame,
    dropped: pd.DataFrame,
) -> pd.DataFrame:
    """Per-(frame, substituted-player) provenance for eligible frames + one row per dropped frame."""
    players = frames[~frames["is_ball"].astype(bool)]
    actual = players[[*RD_FRAME_KEYS, "team_id", "player_id", "x", "y"]].rename(
        columns={"x": "actual_x", "y": "actual_y"}
    )

    # served rows in ELIGIBLE frames, with a finite ghost -> scored; carry actual_x/y for displacement.
    elig_keys = eligible[RD_FRAME_KEYS]
    left, right = align_join_keys(served, elig_keys, list(RD_FRAME_KEYS))
    scored = left.merge(right, on=RD_FRAME_KEYS, how="inner")
    if len(scored):
        left, right = align_join_keys(scored, actual, [*RD_FRAME_KEYS, "team_id", "player_id"])
        scored = left.merge(right, on=[*RD_FRAME_KEYS, "team_id", "player_id"], how="left")
        finite = scored["ghost_x"].notna() & scored["ghost_y"].notna()
        scored = scored[finite].reset_index(drop=True)

    if len(scored):
        scored["displacement_m"] = np.hypot(
            scored["ghost_x"].to_numpy(dtype=float) - scored["actual_x"].to_numpy(dtype=float),
            scored["ghost_y"].to_numpy(dtype=float) - scored["actual_y"].to_numpy(dtype=float),
        )
        scored["drop_reason"] = pd.Series(pd.NA, index=scored.index, dtype="object")

    # eligible frames with NO finite served ghost -> no_ghost_served drop (counted).
    served_frame_keys = set(
        zip(
            _idk(scored["game_id"]) if len(scored) else [],
            _idk(scored["period_id"]) if len(scored) else [],
            _idk(scored["frame_id"]) if len(scored) else [],
            strict=True,
        )
    )
    unserved_mask = np.array(
        [
            (g, p, f) not in served_frame_keys
            for g, p, f in zip(
                _idk(eligible["game_id"]), _idk(eligible["period_id"]), _idk(eligible["frame_id"]), strict=True
            )
        ],
        dtype=bool,
    )
    unserved = eligible.loc[unserved_mask, RD_FRAME_KEYS].copy()
    unserved["drop_reason"] = _DROP_NO_GHOST

    drop_rows = pd.concat([dropped, unserved], ignore_index=True) if len(unserved) else dropped
    out = pd.concat([scored, drop_rows], ignore_index=True) if len(scored) else drop_rows.copy()
    for col in _PROVENANCE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[_PROVENANCE_COLUMNS].reset_index(drop=True)


def _idk(values) -> list:
    return [canonical_id(v) for v in values]


def _write_back(frames: pd.DataFrame, *, provenance: pd.DataFrame) -> pd.DataFrame:
    """Substitute the ghost for A's scored rearguard/keeper rows. PURE -- returns a new frame."""
    out = frames.copy()
    scored = provenance[provenance["drop_reason"].isna()]
    if not len(scored):
        return out
    non_ball = ~out["is_ball"].astype(bool)
    side = out.loc[non_ball, [*RD_FRAME_KEYS, "team_id", "player_id"]]
    ghost = scored[[*RD_FRAME_KEYS, "team_id", "player_id", "ghost_x", "ghost_y"]]
    left, right = align_join_keys(side, ghost, [*RD_FRAME_KEYS, "team_id", "player_id"])
    joined = left.merge(right, on=[*RD_FRAME_KEYS, "team_id", "player_id"], how="left")
    joined.index = side.index
    hit = joined["ghost_x"].notna().to_numpy() & joined["ghost_y"].notna().to_numpy()
    idx = joined.index[hit]
    if len(idx):
        out.loc[idx, "x"] = joined.loc[idx, "ghost_x"].to_numpy(dtype=float)
        out.loc[idx, "y"] = joined.loc[idx, "ghost_y"].to_numpy(dtype=float)
    return out
