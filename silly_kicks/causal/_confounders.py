"""Layer 2's tracking-confounder join (TF-19 sign-off package §5.1, D5).

PROVENANCE IS REGISTERED: every tracking confounder is computed FRESH from frames. It must NOT be
sourced from ``fct_action_context`` -- ADR-045/4.55.0 fixed ``pressure_on_actor__bekkers_pi`` (the
away-team velocity re-projection defect; away values changed, home byte-identical) and the lakehouse
re-materialization of that column is still an OPEN owner action. A mart-sourced join would silently
hand Layer 2's design pre-fix away-team pressure, in a confounder chosen precisely because it is
load-bearing, and no test in this package would notice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: The only accepted provenance. `join_layer2_confounders` refuses anything else by name.
CONFOUNDER_SOURCE = "frames_computed"


def _defending_team_id(spells: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """The team NOT in possession, per spell -- ADR-019 dtype-safe (never a raw ``!=``)."""
    from silly_kicks.id_compat import same_id

    non_ball = frames.loc[~frames["is_ball"].astype(bool), ["game_id", "team_id"]].dropna()
    # Precomputed once per game rather than rescanned per spell: the per-row form was
    # O(n_spells x n_frames) on a frames table, and pandas-stubs types `.loc[mask, col]` as a
    # scalar union that no amount of casting reads cleanly.
    teams_by_game: dict[object, list] = {
        gid: list(grp["team_id"].unique()) for gid, grp in non_ball.groupby("game_id", sort=False)
    }
    out: list[object] = []
    for gid, poss in zip(spells["game_id"], spells["possessing_team"], strict=True):
        other = [t for t in teams_by_game.get(gid, []) if not same_id(t, poss)]
        out.append(other[0] if other else np.nan)
    return pd.Series(out, index=spells.index)


def _pressure_at_entry(spells: pd.DataFrame, frames: pd.DataFrame, add_pressure_on_actor) -> np.ndarray:
    """Carrier pressure at each spell's ENTRY frame, via a synthesized one-row action per spell.

    `add_pressure_on_actor` answers a per-action question; the spell row is frame-anchored. Rather
    than invent a spell-level pressure primitive, the entry is expressed as an action at the entry
    ball position so the SHIPPED, pinned `bekkers_pi` implementation is what runs (spec §6.4 pins
    the measure; ADR-036 pins it again after the 3-method audit).
    """
    from silly_kicks.id_compat import same_id
    from silly_kicks.spadl import config as _cfg

    ball = frames.loc[frames["is_ball"].astype(bool), ["game_id", "period_id", "frame_id", "x", "y"]]
    ball = ball.drop_duplicates(subset=["game_id", "period_id", "frame_id"]).set_index(
        ["game_id", "period_id", "frame_id"]
    )
    idx = pd.MultiIndex.from_arrays([spells["game_id"], spells["period_id"], spells["entry_frame_id"]])
    bx = ball["x"].reindex(idx).to_numpy(dtype=float)
    by = ball["y"].reindex(idx).to_numpy(dtype=float)

    # `_resolve_action_frame_context` locates the ACTOR by `player_id`, so the synthesized action
    # needs one: the possessing-team outfielder nearest the ball at entry, i.e. the carrier by the
    # ordinary definition. Resolved here rather than re-running carrier inference, which
    # `build_opportunities` has already paid for internally and does not surface on the spell row.
    players = frames.loc[~frames["is_ball"].astype(bool)]
    carriers: list[object] = []
    for gid, per, fid, team, cx, cy in zip(
        spells["game_id"],
        spells["period_id"],
        spells["entry_frame_id"],
        spells["possessing_team"],
        bx,
        by,
        strict=True,
    ):
        grp = players[(players["game_id"] == gid) & (players["period_id"] == per) & (players["frame_id"] == fid)]
        grp = grp[[same_id(t, team) for t in grp["team_id"]]]
        if grp.empty or not np.isfinite(cx):
            carriers.append(pd.NA)
            continue
        d2 = (grp["x"].to_numpy(dtype=float) - cx) ** 2 + (grp["y"].to_numpy(dtype=float) - cy) ** 2
        carriers.append(grp["player_id"].to_numpy()[int(np.argmin(d2))])

    synth = pd.DataFrame(
        {
            "game_id": spells["game_id"].to_numpy(),
            "action_id": np.arange(len(spells)),
            "period_id": spells["period_id"].to_numpy(),
            "team_id": spells["possessing_team"].to_numpy(),
            "player_id": pd.array(carriers),
            "time_seconds": spells["entry_time"].to_numpy(dtype=float),
            "type_id": _cfg.actiontype_id["pass"],
            "result_id": _cfg.result_id["success"],
            "start_x": bx,
            "start_y": by,
            "end_x": bx,
            "end_y": by,
        }
    )
    pressed = add_pressure_on_actor(synth, frames, methods=("bekkers_pi",))
    return pressed["pressure_on_actor__bekkers_pi"].to_numpy(dtype=float)


def _time_remaining(spells: pd.DataFrame) -> np.ndarray:
    """Seconds left in the period, from the period's own observed maximum ``end_time``."""
    end = spells.groupby(["game_id", "period_id"])["end_time"].transform("max")
    return (end - spells["entry_time"]).to_numpy(dtype=float)


def join_layer2_confounders(
    spells: pd.DataFrame,
    *,
    frames: pd.DataFrame | None,
    actions: pd.DataFrame | None,
    home_team_id,
    source: str = CONFOUNDER_SOURCE,
) -> pd.DataFrame:
    """Return a NEW spells frame with Layer 2's tracking confounders attached at spell entry.

    Never mutates ``spells``. Raises rather than NaN-filling a missing column: a NaN column looks
    tolerant here and then dies inside ``fit_propensity`` as "Input X contains NaN", naming nothing.
    """
    if source != CONFOUNDER_SOURCE:
        raise ValueError(
            f"Layer 2 confounders must be {CONFOUNDER_SOURCE!r}, got {source!r}. "
            "fct_action_context.pressure_on_actor__bekkers_pi is pre-ADR-045 for away teams until "
            "the lakehouse re-materializes; a mart-sourced join would silently use stale pressure."
        )
    # PRECONDITIONS FIRST, before any join work. `score_differential` is emitted by `_row` when
    # `emit_outcome_partition` is set; it is never NaN-filled here, because a NaN column looks
    # tolerant and then raises inside `fit_propensity` as "Input X contains NaN" (MEASURED), naming
    # nothing, deep in the run.
    if "score_differential" not in spells.columns:
        raise ValueError(
            "score_differential absent from spells: build them with a config that emits it "
            "(layer2_config), or the design matrix will fail inside fit_propensity."
        )

    # Ordered AFTER the two guards above so a malformed call still reports the most specific cause.
    if frames is None or actions is None:
        raise ValueError(
            "join_layer2_confounders needs both `frames` and `actions`: every tracking confounder is "
            f"computed fresh ({CONFOUNDER_SOURCE}), so there is nothing to compute them from."
        )

    from silly_kicks.tracking import add_pressure_on_actor, compute_defensive_line, resolve_defended_goals

    out = spells.copy()

    # SIGNATURES VERIFIED BY EXECUTION -- do not "simplify" either call:
    #   compute_defensive_line(frames, *, goal_map, n=4, adaptive_max_n=5)
    #   add_pressure_on_actor(actions, frames, *, links=None, methods=(...), ...)  <- methods is
    #   PLURAL and a TUPLE; a singular `method="bekkers_pi"` raises TypeError.
    # ADR-051 D3: direction comes from the goal map, built ONCE from these frames. This is a
    # CAUSAL COVARIATE path -- if the values move, docs/research/covariate_invariance/ is stale.
    line = compute_defensive_line(frames, goal_map=resolve_defended_goals(frames))
    # Columns: game_id, period_id, frame_id, team_id, defensive_line_x, back_line_high_x,
    # compactness_x, lateral_width, max_lateral_gap. There is NO `defensive_line_spread`.
    # The key has FOUR levels and the function computes for BOTH teams, so the DEFENDING team must
    # be selected -- Layer 2's confounder is the line the attacking spell FACES. Joining on three
    # levels would silently pick an arbitrary team's row.
    keyed = line.set_index(["game_id", "period_id", "frame_id", "team_id"])
    idx = pd.MultiIndex.from_arrays(
        [out["game_id"], out["period_id"], out["entry_frame_id"], _defending_team_id(out, frames)]
    )
    out["defensive_line_height"] = keyed["defensive_line_x"].reindex(idx).to_numpy()
    out["defensive_line_compactness"] = keyed["compactness_x"].reindex(idx).to_numpy()

    # Pressure is a per-ACTION quantity, but a spell row is FRAME-anchored -- it carries
    # `entry_frame_id`, never an action id. Joining on a non-existent `entry_action_id` would have
    # left this confounder structurally all-NaN, which `build_design_matrix` then aborts the run on.
    # So the spell's entry is expressed as a one-row synthesized action at the entry ball position
    # (the same synthesize-then-reuse idiom the atomic mirrors use), and the shipped aggregator is
    # asked the question it actually answers: pressure on the carrier at that moment.
    out["pressure_on_actor__bekkers_pi"] = _pressure_at_entry(out, frames, add_pressure_on_actor)

    out["time_remaining_s"] = _time_remaining(out)
    return out
