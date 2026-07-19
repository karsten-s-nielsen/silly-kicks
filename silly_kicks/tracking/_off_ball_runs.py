"""Off-ball runs + line-break kernels (TF-4).

Novel implementations inspired by the OBSO framework (Spearman 2018).
Off-ball-runs: per-attacking-teammate displacement in the pre-action window.
Line-break: action destination vs opposing team's defensive line geometry.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.id_compat import (
    align_join_keys,
    canonical_id,
    canonical_id_series,
    ids_differ,
    ids_equal,
    same_id,
)

from ._action_orientation import acting_team_attacks_rtl, validate_period_directions

_OFF_BALL_RUNS_COLS = [
    "n_off_ball_runners_pre_window",
    "max_off_ball_run_displacement_pre_window",
    "mean_off_ball_run_speed_pre_window",
    "n_off_ball_runners_toward_goal_pre_window",
]

_LINE_BREAK_COLS = [
    "line_break",
    "n_attackers_behind_line",
]


def _validate_ltr(frames: pd.DataFrame) -> None:
    """Raise ValueError if frames are not period-normalized (home attacks LTR).

    After ``play_left_to_right``, home-team rows have ``"ltr"`` and away-team
    rows have ``"rtl"`` — both valid in the period-normalized frame. Rejects
    frames with unexpected direction values or frames with only ``"rtl"``
    (period normalization not applied).
    """
    if "team_attacking_direction" in frames.columns:
        directions = set(frames["team_attacking_direction"].dropna().unique())
        valid = {"ltr", "rtl"}
        unexpected = directions - valid
        if unexpected:
            raise ValueError(
                "_off_ball_runs: frames have unexpected team_attacking_direction "
                f"values: {sorted(unexpected)}. Expected 'ltr'/'rtl' only."
            )
        if directions and "ltr" not in directions:
            raise ValueError(
                "_off_ball_runs: frames must be period-normalized "
                "(play_left_to_right). Found only 'rtl' direction values — "
                "no home-team rows with 'ltr'."
            )
    # ADR-041: reject only a team that CONTRADICTS ITSELF (both directions in one period).
    # A uniform or all-null column is deliberately accepted -- it means unoriented, or a
    # different convention (snapshot frames are already action-LTR), not an error.
    validate_period_directions(frames, caller="_off_ball_runs")


def _prepare_run_candidates(sliced: pd.DataFrame) -> pd.DataFrame:
    """Shared TF-4 / TF-35 off-ball-run CANDIDACY (ADR-042; 3rd-consumer extraction).

    Keeps same-team-as-actor, non-actor, non-goalkeeper rows; drops NaN positions and
    dead-ball frames. Expects ``sliced`` to already carry ``action_team_id`` /
    ``actor_player_id`` (the actor merge) and to have had ball rows removed.

    Action-level dead-ball tagging and the ``<2 frames`` skip stay loop-local in each
    consumer -- only this leaf predicate is shared, which is what the identity gate
    between the two implementations then has to sentinel.

    Extracted rather than duplicated because ADR-041/042 makes this the THIRD consumer
    (the TF-4 kernel, TF-35's ``detect_off_ball_runs``, and the gate that compares them),
    and the house rule triggers extraction at three.
    """
    teammates = sliced[
        ids_equal(sliced["team_id"], sliced["action_team_id"]).to_numpy()
        & ids_differ(sliced["player_id"], sliced["actor_player_id"]).to_numpy()
        & (~sliced["is_goalkeeper"].astype(bool)).to_numpy()
    ].copy()
    teammates = teammates.dropna(subset=["x", "y"])
    if "ball_state" in teammates.columns:
        teammates = teammates[teammates["ball_state"] != "dead"]
    return teammates


def _off_ball_runs_kernel(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    pre_seconds: float = 1.5,
    min_displacement_m: float = 3.0,
) -> pd.DataFrame:
    """Per-action off-ball-run metrics for attacking teammates.

    Returns DataFrame aligned with actions.index, columns: _OFF_BALL_RUNS_COLS.
    """
    from .utils import slice_around_event

    n_actions = len(actions)
    empty = pd.DataFrame(
        {
            "n_off_ball_runners_pre_window": pd.array([pd.NA] * n_actions, dtype="Int64"),
            "max_off_ball_run_displacement_pre_window": np.full(n_actions, np.nan),
            "mean_off_ball_run_speed_pre_window": np.full(n_actions, np.nan),
            "n_off_ball_runners_toward_goal_pre_window": pd.array([pd.NA] * n_actions, dtype="Int64"),
        },
        index=actions.index,
    )

    if n_actions == 0 or len(frames) == 0:
        return empty

    _validate_ltr(frames)

    # Partition by game_id to prevent period_id collisions across games
    results = []
    for game_id, game_actions in actions.groupby("game_id", sort=False):
        game_frames = frames[frames["game_id"] == game_id]
        if game_frames.empty:
            results.append(empty.reindex(game_actions.index))
            continue

        sliced = slice_around_event(game_actions, game_frames, pre_seconds=pre_seconds, post_seconds=0.0)
        if sliced.empty:
            results.append(empty.reindex(game_actions.index))
            continue

        # Extract dead-ball state from sliced BEFORE removing ball rows
        action_end_ball_state: dict = {}
        if "ball_state" in game_frames.columns:
            ball_in_window = sliced[sliced["is_ball"].astype(bool)]
            if not ball_in_window.empty:
                abs_offset = ball_in_window["time_offset_seconds"].abs()
                closest_idx = abs_offset.groupby(ball_in_window["action_id"]).idxmin()
                closest = ball_in_window.loc[closest_idx]
                dead_actions = closest.loc[closest["ball_state"] == "dead", "action_id"]
                action_end_ball_state = {aid: "dead" for aid in dead_actions}

        # NOW remove ball rows
        sliced = sliced[~sliced["is_ball"].astype(bool)].copy()

        # Build per-action teammate displacements
        actor_id_per_action = game_actions[["action_id", "player_id", "team_id"]].rename(
            columns={"player_id": "actor_player_id", "team_id": "action_team_id"}
        )
        sliced = sliced.merge(actor_id_per_action, on="action_id", how="left")

        teammates = _prepare_run_candidates(sliced)

        # Compute per (action_id, player_id): first and last position
        # Use positional arrays to avoid .at assignment issues with Int64 dtype
        n_game = len(game_actions)
        n_runners_arr = np.full(n_game, np.nan)
        max_disp_arr = np.full(n_game, np.nan)
        mean_speed_arr = np.full(n_game, np.nan)
        toward_goal_arr = np.full(n_game, np.nan)

        # Build positional lookup: action_id -> positional index in game_actions
        aid_to_pos = {aid: pos for pos, aid in enumerate(game_actions["action_id"].values)}
        # Single orientation authority (ADR-028 / ADR-041), keyed by action_id: the frames'
        # own team_attacking_direction. Safe to rely on now that validate_period_directions
        # rejects mislabelled frames -- before that guard, uniform-"ltr" frames made this
        # silently report "no flip" for the away team.
        flip_by_action = dict(
            zip(
                game_actions["action_id"].to_numpy(),
                acting_team_attacks_rtl(game_actions, game_frames).to_numpy(dtype=bool),
                strict=False,
            )
        )

        # Actions that appear in the sliced data are "linked" — if they have no
        # teammates in the groupby they should get 0 runners (not NaN).
        linked_aids = set(sliced["action_id"].unique())
        for aid in linked_aids:
            if action_end_ball_state.get(aid) == "dead":
                continue
            if aid in aid_to_pos:
                pos = aid_to_pos[aid]
                n_runners_arr[pos] = 0
                toward_goal_arr[pos] = 0

        for aid, action_group in teammates.groupby("action_id", sort=False):
            # Check dead-ball at action time
            if action_end_ball_state.get(aid) == "dead":
                continue  # stays NaN

            if aid not in aid_to_pos:
                continue
            pos = aid_to_pos[aid]

            # Toward-goal from the SINGLE orientation authority (ADR-028 / ADR-041): the
            # frames' own team_attacking_direction, not home/away identity.
            #
            # HONEST FRAMING: this buys CONSISTENCY, not correctness. On correctly-labelled
            # frames the two authorities already agree; on UNORIENTED frames (all-null
            # direction) both are arbitrary -- identity-keying flips the away team on a
            # frame that never claimed an orientation, and this returns False. TF-4 was the
            # last module keyed on identity while acting_team_attacks_rtl already had 7
            # production call sites, so aligning it removes a divergence; it does not fix
            # wrong numbers. Behaviour on unoriented frames is pinned by
            # test_off_ball_runs_orientation.py so it is documented, not incidental.
            attacks_rtl = bool(flip_by_action.get(aid, False))

            per_player = action_group.sort_values("time_seconds").groupby("player_id", sort=False)

            runners = 0
            toward_goal = 0
            displacements: list[float] = []

            for _pid, player_frames in per_player:
                if len(player_frames) < 2:
                    continue
                x_start = float(player_frames["x"].iloc[0])
                y_start = float(player_frames["y"].iloc[0])
                x_end = float(player_frames["x"].iloc[-1])
                y_end = float(player_frames["y"].iloc[-1])
                disp = float(np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2))
                if disp >= min_displacement_m:
                    runners += 1
                    displacements.append(disp)
                    # dx in the acting team's OWN attacking sense (action-LTR).
                    dx = -(x_end - x_start) if attacks_rtl else (x_end - x_start)
                    if dx > 0:
                        toward_goal += 1

            n_runners_arr[pos] = runners
            toward_goal_arr[pos] = toward_goal
            if displacements:
                max_disp_arr[pos] = max(displacements)
                # Note: this is mean(displacement) / window_duration, not
                # mean(displacement_i / observed_duration_i). For continuous
                # tracking data (all players visible throughout) the two are
                # equivalent. The denominator is the fixed window, making the
                # metric a "displacement rate across the pre-window."
                mean_speed_arr[pos] = float(np.mean(displacements)) / pre_seconds

        game_out = pd.DataFrame(
            {
                "n_off_ball_runners_pre_window": pd.array(
                    [pd.NA if np.isnan(v) else int(v) for v in n_runners_arr], dtype="Int64"
                ),
                "max_off_ball_run_displacement_pre_window": max_disp_arr,
                "mean_off_ball_run_speed_pre_window": mean_speed_arr,
                "n_off_ball_runners_toward_goal_pre_window": pd.array(
                    [pd.NA if np.isnan(v) else int(v) for v in toward_goal_arr], dtype="Int64"
                ),
            },
            index=game_actions.index,
        )
        results.append(game_out)

    if not results:
        return empty

    return pd.concat(results).loc[actions.index]


def _line_break_kernel(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    n: int = 4,
    links: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-action line-break detection and attacker count behind defensive line.

    Returns DataFrame aligned with actions.index, columns: _LINE_BREAK_COLS.
    """
    from ._defensive_line import compute_defensive_line
    from .utils import link_actions_to_frames

    n_actions = len(actions)
    empty = pd.DataFrame(
        {
            "line_break": pd.array([pd.NA] * n_actions, dtype="boolean"),
            "n_attackers_behind_line": pd.array([pd.NA] * n_actions, dtype="Int64"),
        },
        index=actions.index,
    )

    if n_actions == 0 or len(frames) == 0:
        return empty

    # Compute defensive line for all frames (ONCE)
    dl = compute_defensive_line(frames, home_team_id=home_team_id, n=n)
    if dl.empty:
        return empty

    # Link actions to frames
    if links is not None:
        pointers = links
    else:
        pointers, _report = link_actions_to_frames(actions, frames)
    linked = pointers[pointers["frame_id"].notna()].copy()
    if linked.empty:
        return empty

    linked["frame_id_int"] = linked["frame_id"].astype("int64")

    # Join with actions to get team_id, end_x, period_id, game_id
    linked = linked.merge(
        actions[["action_id", "team_id", "end_x", "period_id", "game_id"]],
        on="action_id",
        how="left",
    )

    # Align game_id dtype between linked (from actions) and dl (from frames)
    # before the merge — pandas rejects merge on object vs int64 keys.
    if len(linked) > 0 and len(dl) > 0:
        linked_gid_dtype = linked["game_id"].dtype
        dl_gid_dtype = dl["game_id"].dtype
        if linked_gid_dtype != dl_gid_dtype:
            linked["game_id"] = linked["game_id"].astype(str)
            dl = dl.copy()
            dl["game_id"] = dl["game_id"].astype(str)

    # Join with defensive-line data: match on (game_id, period_id, frame_id)
    # Align id-valued join keys (incl. the frame_id_int<->frame_id pair) so a string-id caller
    # does not raise on the merge (ADR-019).
    linked, dl = align_join_keys(linked, dl, ["game_id", "period_id", ("frame_id_int", "frame_id")])
    merged = linked.merge(
        dl,
        left_on=["game_id", "period_id", "frame_id_int"],
        right_on=["game_id", "period_id", "frame_id"],
        how="left",
        suffixes=("_action", "_dl"),
    )
    # Keep only rows where dl team != action team (opposing team's line)
    opposing = merged[ids_differ(merged["team_id_dl"], merged["team_id_action"])].copy()
    opposing = opposing.drop_duplicates("action_id", keep="first")

    # Use positional arrays to avoid .at assignment issues with nullable dtypes
    n_act = len(actions)
    line_break_arr = np.full(n_act, np.nan)  # will map to boolean
    n_behind_arr = np.full(n_act, np.nan)  # will map to Int64

    # Build positional lookup: action_id -> positional index in actions
    aid_to_pos = {aid: pos for pos, aid in enumerate(actions["action_id"].values)}

    # Pre-build grouped dict for O(1) frame-player lookups. Canonicalize the id-valued group
    # keys (game_id, team_id) so the lookup matches the action-side key regardless of caller
    # dtype (ADR-019; replaces the prior game_id-only isinstance/astype workaround).
    non_ball_non_gk = frames[(~frames["is_ball"].astype(bool)) & (~frames["is_goalkeeper"].astype(bool))].copy()
    non_ball_non_gk["_gid_key"] = canonical_id_series(non_ball_non_gk["game_id"])
    non_ball_non_gk["_team_key"] = canonical_id_series(non_ball_non_gk["team_id"])
    frame_groups: dict = dict(
        iter(non_ball_non_gk.groupby(["_gid_key", "period_id", "frame_id", "_team_key"], sort=False))
    )

    for _, row in opposing.iterrows():
        aid = row["action_id"]
        if aid not in aid_to_pos:
            continue
        pos = aid_to_pos[aid]

        def_line_x = row["defensive_line_x"]
        if pd.isna(def_line_x):
            continue  # stays pd.NA

        action_team = row["team_id_action"]
        end_x = row["end_x"]

        # Coordinate-frame resolution
        if same_id(action_team, home_team_id):
            spadl_def_line_x = def_line_x
        else:
            spadl_def_line_x = 105.0 - def_line_x

        # Line-break: end_x > spadl_def_line_x (both in action-team SPADL frame)
        line_break_arr[pos] = 1.0 if end_x > spadl_def_line_x else 0.0

        # Count attackers behind line — O(1) lookup from pre-built dict
        frame_id = int(row["frame_id_int"])
        period_id = row["period_id"]
        game_id_val = row["game_id"]
        key = (canonical_id(game_id_val), period_id, frame_id, canonical_id(action_team))
        frame_players = frame_groups.get(key, pd.DataFrame())

        if frame_players.empty:
            n_behind_arr[pos] = 0
            continue

        # In tracking coords:
        # Home-team attackers "behind" away line: tracking x > defensive_line_x
        # Away-team attackers "behind" home line: tracking x < defensive_line_x
        if same_id(action_team, home_team_id):
            behind_mask = frame_players["x"] > def_line_x
        else:
            behind_mask = frame_players["x"] < def_line_x

        n_behind_arr[pos] = int(behind_mask.sum())

    return pd.DataFrame(
        {
            "line_break": pd.array([pd.NA if np.isnan(v) else bool(v) for v in line_break_arr], dtype="boolean"),
            "n_attackers_behind_line": pd.array(
                [pd.NA if np.isnan(v) else int(v) for v in n_behind_arr], dtype="Int64"
            ),
        },
        index=actions.index,
    )
