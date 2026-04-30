"""End-to-end walkthrough: PFF FC WC 2022 -> SPADL -> Atomic-SPADL -> VAEP labels.

This script is **documentation, not test**. It runs against a user-supplied
local PFF directory and demonstrates how to use the silly-kicks public API.

Usage:
    uv run python docs/examples/pff_wc2022_walkthrough.py /path/to/PFF/WC2022/

The PFF directory is expected to contain:
- `Event Data/<match_id>.json`
- `Metadata/<match_id>.json`
- `Rosters/<match_id>.json`

This file is excluded from automated test discovery (it lives outside
``tests/``) and is not invoked by pytest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from silly_kicks.atomic.spadl import convert_to_atomic
from silly_kicks.spadl import add_names, boundary_metrics, coverage_metrics, pff
from silly_kicks.vaep.labels import scores

# Match 10502 = NED-USA, 2022 R16. Edit to point at any match in the directory.
DEFAULT_MATCH_ID = 10502


def load_pff_events(pff_dir: Path, match_id: int) -> tuple[pd.DataFrame, dict]:
    """Load and flatten one PFF match's event data into the
    :data:`silly_kicks.spadl.pff.EXPECTED_INPUT_COLUMNS` shape.

    Returns
    -------
    (events_df, metadata)
        events_df : pd.DataFrame
            Flat DataFrame matching ``pff.EXPECTED_INPUT_COLUMNS``.
        metadata : dict
            Match metadata extracted from ``Metadata/<match_id>.json``
            (used to populate ``home_team_id`` and ``home_team_start_left``).
    """
    with (pff_dir / "Event Data" / f"{match_id}.json").open("r", encoding="utf-8") as f:
        events_json = json.load(f)
    with (pff_dir / "Metadata" / f"{match_id}.json").open("r", encoding="utf-8") as f:
        meta_list = json.load(f)
    metadata = meta_list[0] if isinstance(meta_list, list) else meta_list

    with (pff_dir / "Rosters" / f"{match_id}.json").open("r", encoding="utf-8") as f:
        roster_json = json.load(f)
    pid_to_team = {int(r["player"]["id"]): int(r["team"]["id"]) for r in roster_json}

    rows = []
    for ev in events_json:
        ge = ev.get("gameEvents") or {}
        pe = ev.get("possessionEvents") or {}
        # Real PFF data carries `fouls` as a single dict per event (not a list).
        f0 = ev.get("fouls") or {}
        ball = (ev.get("ball") or [{}])[0] if ev.get("ball") else {}

        challenger_pid = pe.get("challengerPlayerId")
        winner_pid = pe.get("challengeWinnerPlayerId")

        rows.append(
            {
                "game_id": int(ev["gameId"]),
                "event_id": int(ev["gameEventId"]),
                "possession_event_id": ev.get("possessionEventId"),
                "period_id": int(ge.get("period") or 0),
                "time_seconds": float(ge.get("startGameClock") or 0.0),
                "team_id": int(ge.get("teamId") or 0),
                "player_id": ge.get("playerId"),
                "game_event_type": ge.get("gameEventType"),
                "possession_event_type": pe.get("possessionEventType"),
                "set_piece_type": ge.get("setpieceType"),
                "ball_x": ball.get("x"),
                "ball_y": ball.get("y"),
                "body_type": pe.get("bodyType"),
                "ball_height_type": pe.get("ballHeightType"),
                "pass_outcome_type": pe.get("passOutcomeType"),
                "pass_type": pe.get("passType"),
                "incompletion_reason_type": pe.get("incompletionReasonType"),
                "cross_outcome_type": pe.get("crossOutcomeType"),
                "cross_type": pe.get("crossType"),
                "cross_zone_type": pe.get("crossZoneType"),
                "shot_outcome_type": pe.get("shotOutcomeType"),
                "shot_type": pe.get("shotType"),
                "shot_nature_type": pe.get("shotNatureType"),
                "shot_initial_height_type": pe.get("shotInitialHeightType"),
                "save_height_type": pe.get("saveHeightType"),
                "save_rebound_type": pe.get("saveReboundType"),
                "carry_type": pe.get("carryType"),
                "ball_carry_outcome": pe.get("ballCarryOutcome"),
                "carry_intent": pe.get("carryIntent"),
                "carry_defender_player_id": pe.get("carryDefenderPlayerId"),
                "challenge_type": pe.get("challengeType"),
                "challenge_outcome_type": pe.get("challengeOutcomeType"),
                "challenger_player_id": challenger_pid,
                "challenger_team_id": pid_to_team.get(int(challenger_pid)) if challenger_pid is not None else None,
                "challenge_winner_player_id": winner_pid,
                "challenge_winner_team_id": pid_to_team.get(int(winner_pid)) if winner_pid is not None else None,
                "tackle_attempt_type": pe.get("tackleAttemptType"),
                "clearance_outcome_type": pe.get("clearanceOutcomeType"),
                "rebound_outcome_type": pe.get("reboundOutcomeType"),
                "keeper_touch_type": pe.get("keeperTouchType"),
                "touch_outcome_type": pe.get("touchOutcomeType"),
                "touch_type": pe.get("touchType"),
                "foul_type": f0.get("foulType"),
                "on_field_offense_type": f0.get("onFieldOffenseType"),
                "final_offense_type": f0.get("finalOffenseType"),
                "on_field_foul_outcome_type": f0.get("onFieldFoulOutcomeType"),
                "final_foul_outcome_type": f0.get("finalFoulOutcomeType"),
            }
        )
    df = pd.DataFrame(rows)

    for col in (
        "possession_event_id",
        "player_id",
        "carry_defender_player_id",
        "challenger_player_id",
        "challenger_team_id",
        "challenge_winner_player_id",
        "challenge_winner_team_id",
    ):
        df[col] = df[col].astype("Int64")

    return df, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pff_dir", type=Path, help="Path to the PFF FC WC 2022 directory")
    parser.add_argument("--match-id", type=int, default=DEFAULT_MATCH_ID)
    args = parser.parse_args()

    print(f"Loading match {args.match_id} from {args.pff_dir}...")
    events, metadata = load_pff_events(args.pff_dir, args.match_id)
    print(f"  Loaded {len(events)} events.")

    home_team_id = int(metadata["homeTeam"]["id"])
    home_team_start_left = bool(metadata["homeTeamStartLeft"])
    home_team_start_left_extratime = (
        bool(metadata["homeTeamStartLeftExtraTime"]) if metadata.get("homeTeamStartLeftExtraTime") is not None else None
    )

    actions, report = pff.convert_to_actions(
        events,
        home_team_id=home_team_id,
        home_team_start_left=home_team_start_left,
        home_team_start_left_extratime=home_team_start_left_extratime,
    )
    print(f"  Converted to {report.total_actions} SPADL actions.")
    if report.has_unrecognized:
        print(f"  WARNING: unrecognized vocabulary: {report.unrecognized_counts}")
    print(f"  Action-type counts: {dict(sorted(report.mapped_counts.items()))}")

    atomic_actions = convert_to_atomic(actions)
    print(f"  Atomic-SPADL: {len(atomic_actions)} atomic actions.")

    cov = coverage_metrics(actions, expected_action_types={"pass", "shot", "tackle", "cross"})
    print(f"  Coverage: {cov}")

    bm = boundary_metrics(actions)
    print(f"  Boundary: {bm}")

    actions_named = add_names(actions)
    s = scores(actions_named, nr_actions=10)
    print(f"  VAEP scores label: {int(s['scores'].sum())} positive rows.")


if __name__ == "__main__":
    main()
