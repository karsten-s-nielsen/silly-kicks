"""Off-ball runs + line-break kernels (TF-4).

Novel implementations inspired by the OBSO framework (Spearman 2018).
Off-ball-runs: per-attacking-teammate displacement in the pre-action window.
Line-break: action destination vs opposing team's defensive line geometry.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
    """Raise ValueError if frames contain non-LTR direction values."""
    if "team_attacking_direction" in frames.columns:
        directions = frames["team_attacking_direction"].dropna().unique()
        non_ltr = [d for d in directions if d != "ltr"]
        if non_ltr:
            raise ValueError(
                "_off_ball_runs: frames must be LTR-normalized "
                "(play_left_to_right). Found non-'ltr' values in "
                f"team_attacking_direction: {non_ltr}"
            )


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

        # Keep only same-team, non-actor, non-goalkeeper teammates
        teammates = sliced[
            (sliced["team_id"] == sliced["action_team_id"])
            & (sliced["player_id"] != sliced["actor_player_id"])
            & (~sliced["is_goalkeeper"].astype(bool))
        ].copy()

        # Drop NaN positions
        teammates = teammates.dropna(subset=["x", "y"])

        # Exclude dead-ball frames within window
        if "ball_state" in teammates.columns:
            teammates = teammates[teammates["ball_state"] != "dead"]

        # Compute per (action_id, player_id): first and last position
        # Use positional arrays to avoid .at assignment issues with Int64 dtype
        n_game = len(game_actions)
        n_runners_arr = np.full(n_game, np.nan)
        max_disp_arr = np.full(n_game, np.nan)
        mean_speed_arr = np.full(n_game, np.nan)
        toward_goal_arr = np.full(n_game, np.nan)

        # Build positional lookup: action_id -> positional index in game_actions
        aid_to_pos = {aid: pos for pos, aid in enumerate(game_actions["action_id"].values)}
        # O(1) team lookup per action
        action_team_lookup = game_actions.set_index("action_id")["team_id"]

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

            # Get action's team_id for toward-goal direction — O(1) lookup
            action_team = action_team_lookup.loc[aid]
            is_home = action_team == home_team_id

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
                    dx = x_end - x_start
                    # Home team: toward goal = positive dx
                    # Away team: toward goal = negative dx
                    if (is_home and dx > 0) or (not is_home and dx < 0):
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

    # Join with defensive-line data: match on (game_id, period_id, frame_id)
    merged = linked.merge(
        dl,
        left_on=["game_id", "period_id", "frame_id_int"],
        right_on=["game_id", "period_id", "frame_id"],
        how="left",
        suffixes=("_action", "_dl"),
    )
    # Keep only rows where dl team != action team (opposing team's line)
    opposing = merged[merged["team_id_dl"] != merged["team_id_action"]].copy()
    opposing = opposing.drop_duplicates("action_id", keep="first")

    # Use positional arrays to avoid .at assignment issues with nullable dtypes
    n_act = len(actions)
    line_break_arr = np.full(n_act, np.nan)  # will map to boolean
    n_behind_arr = np.full(n_act, np.nan)  # will map to Int64

    # Build positional lookup: action_id -> positional index in actions
    aid_to_pos = {aid: pos for pos, aid in enumerate(actions["action_id"].values)}

    # Pre-build grouped dict for O(1) frame-player lookups
    non_ball_non_gk = frames[(~frames["is_ball"].astype(bool)) & (~frames["is_goalkeeper"].astype(bool))]
    frame_groups: dict = dict(
        iter(non_ball_non_gk.groupby(["game_id", "period_id", "frame_id", "team_id"], sort=False))
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
        if action_team == home_team_id:
            spadl_def_line_x = def_line_x
        else:
            spadl_def_line_x = 105.0 - def_line_x

        # Line-break: end_x > spadl_def_line_x (both in action-team SPADL frame)
        line_break_arr[pos] = 1.0 if end_x > spadl_def_line_x else 0.0

        # Count attackers behind line — O(1) lookup from pre-built dict
        frame_id = int(row["frame_id_int"])
        period_id = row["period_id"]
        game_id_val = row["game_id"]
        key = (game_id_val, period_id, frame_id, action_team)
        frame_players = frame_groups.get(key, pd.DataFrame())

        if frame_players.empty:
            n_behind_arr[pos] = 0
            continue

        # In tracking coords:
        # Home-team attackers "behind" away line: tracking x > defensive_line_x
        # Away-team attackers "behind" home line: tracking x < defensive_line_x
        if action_team == home_team_id:
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
