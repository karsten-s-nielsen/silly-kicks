"""End-to-end walkthrough: Gradient Sports (PFF FC) WC 2022 -> SPADL -> Atomic-SPADL -> VAEP labels.

This script is **documentation, not test**. It runs against a user-supplied
local Gradient Sports (PFF) directory and demonstrates how to use the silly-kicks public API.

Usage:
    uv run python docs/examples/gradientsports_wc2022_walkthrough.py /path/to/GradientSports/WC2022/

The data directory is expected to contain:
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
from silly_kicks.spadl import add_names, add_possessions, boundary_metrics, coverage_metrics, gradientsports
from silly_kicks.vaep.labels import scores

# Match 10502 = NED-USA, 2022 R16. Edit to point at any match in the directory.
DEFAULT_MATCH_ID = 10502


def load_gradientsports_events(gs_dir: Path, match_id: int) -> tuple[pd.DataFrame, dict]:
    """Load and flatten one Gradient Sports (PFF) match's event data into the
    :data:`silly_kicks.spadl.gradientsports.EXPECTED_INPUT_COLUMNS` shape.

    Returns
    -------
    (events_df, metadata)
        events_df : pd.DataFrame
            Flat DataFrame matching ``gradientsports.EXPECTED_INPUT_COLUMNS``.
        metadata : dict
            Match metadata extracted from ``Metadata/<match_id>.json``
            (used to populate ``home_team_id`` and ``home_team_start_left``).
    """
    with (gs_dir / "Event Data" / f"{match_id}.json").open("r", encoding="utf-8") as f:
        events_json = json.load(f)
    with (gs_dir / "Metadata" / f"{match_id}.json").open("r", encoding="utf-8") as f:
        meta_list = json.load(f)
    metadata = meta_list[0] if isinstance(meta_list, list) else meta_list

    with (gs_dir / "Rosters" / f"{match_id}.json").open("r", encoding="utf-8") as f:
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
                # The game event's start (the kickoff anchor and sort tiebreak) and the possession event's own time;
                # the converter's time_seconds is event_time minus the period's kickoff start_time (ADR-133).
                "start_time": ev.get("startTime"),
                "event_time": ev.get("eventTime"),
                # PFF's native in-play sequence: one stretch of play from a restart to the next stoppage. It can hold
                # several team possessions. Kept through the converter with preserve_native for the boundary check.
                "sequence": ev.get("sequence"),
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
    parser.add_argument("gs_dir", type=Path, help="Path to the Gradient Sports (PFF FC) WC 2022 directory")
    parser.add_argument("--match-id", type=int, default=DEFAULT_MATCH_ID)
    args = parser.parse_args()

    print(f"Loading match {args.match_id} from {args.gs_dir}...")
    events, metadata = load_gradientsports_events(args.gs_dir, args.match_id)
    print(f"  Loaded {len(events)} events.")

    home_team_id = int(metadata["homeTeam"]["id"])
    home_team_start_left = bool(metadata["homeTeamStartLeft"])
    home_team_start_left_extratime = (
        bool(metadata["homeTeamStartLeftExtraTime"]) if metadata.get("homeTeamStartLeftExtraTime") is not None else None
    )

    actions, report = gradientsports.convert_to_actions(
        events,
        home_team_id=home_team_id,
        home_team_start_left=home_team_start_left,
        home_team_start_left_extratime=home_team_start_left_extratime,
        preserve_native=["sequence"],
    )
    print(f"  Converted to {report.total_actions} SPADL actions.")
    if report.has_unrecognized:
        print(f"  WARNING: unrecognized vocabulary: {report.unrecognized_counts}")
    print(f"  Action-type counts: {dict(sorted(report.mapped_counts.items()))}")

    atomic_actions = convert_to_atomic(actions)
    print(f"  Atomic-SPADL: {len(atomic_actions)} atomic actions.")

    expected = {"pass", "shot", "tackle", "cross"}
    cov = coverage_metrics(actions=actions, expected_action_types=expected)
    print(
        f"  Coverage: {cov['total_actions']} actions; expected types missing: {cov['missing'] or 'none'} "
        f"(a full match should miss none of {sorted(expected)})."
    )

    # Possession boundaries: add_possessions' heuristic team possessions vs PFF's native in-play `sequence` (one
    # stretch of play from a restart to the next stoppage), on the rows that carry a sequence. The two split play
    # differently: PFF at every stoppage, the heuristic at changes of team possession. So neither number reaches
    # 1.0 on clean data. Recall = the share of PFF's sequence starts the heuristic also marks; precision = the share
    # of heuristic boundaries that are also sequence starts (one sequence often holds several possessions).
    # Reference, all 64 WC2022 matches: recall 0.80-0.91 (median 0.86), precision 0.46-0.64 (median 0.58). A match
    # far below these ranges points to a conversion or input problem.
    with_possessions = add_possessions(actions)
    known = with_possessions["sequence"].notna()
    bm = boundary_metrics(
        heuristic=with_possessions.loc[known, "possession_id"],
        native=with_possessions.loc[known, "sequence"].astype("int64"),
    )
    print(
        f"  Possession boundaries vs PFF sequences: recall {bm['recall']:.2f} (WC2022 range 0.80-0.91), "
        f"precision {bm['precision']:.2f} (WC2022 range 0.46-0.64)."
    )

    actions_named = add_names(actions)
    s = scores(actions_named, nr_actions=10)
    print(f"  VAEP scores label: {int(s['scores'].sum())} positive rows.")


if __name__ == "__main__":
    main()
