"""SkillCorner events-only goals reconciliation (TF-46 Arm 1, spec section 6).

SkillCorner Game Intelligence records goals unreliably: a goal may carry no shot row, a shot row may
carry the wrong scorer, and an own goal is inverted or unsignalled. Ground truth is ``match.json``: the
per-player ``goal`` / ``own_goal`` counts and the team scores. The GI running score (``team_score`` /
``opponent_team_score`` on each possession) places each goal in time and names the scoring team.

``reconcile_goals`` runs the two-pass repair (spec section 6.2). Each repaired or inferred row carries a
visible marker: ``result_source="inferred"`` and a ``repair_kind`` naming the class (decision A2). The
ambiguous classes and a ``match.json`` coverage gap are reported, never silently repaired.

Pass 1 (over score changes): ``normal`` / ``scored_tag`` / ``own_goal_inverted``.
Pass 2 (unexplained changes): ``own_goal_surplus`` / ``swap`` / ``missing_goal`` / ``own_goal_unsignalled``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import cast

import numpy as np
import pandas as pd

from ..id_compat import canonical_id, same_id
from . import config as spadlconfig


def _cid(value: object) -> str:
    """Canonical id as a plain ``str`` (ids here are non-null by construction, so never ``<NA>``)."""
    return str(canonical_id(value))


#: The conceding team's own-goal end point: its own goal-mouth centre. Each SkillCorner possession is in
#: its own attacking-left-to-right frame (x toward the opponent goal at field_length), so the team's own
#: goal is at x = 0 (spec section 6.3).
_OWN_GOAL_END = (0.0, spadlconfig.field_width / 2.0)

#: The constant penalty spot in the taking team's attacking-left-to-right frame (spec section 9.3).
_PENALTY_SPOT = (94.0, 34.0)


@dataclasses.dataclass(frozen=True)
class GoalsReport:
    """Summary of the goals reconciliation (attached to the SkillCorner conversion report)."""

    repaired: Mapping[str, int]
    ambiguous: int
    coverage_gap: bool
    unresolved: int
    final_home_goals: int
    final_away_goals: int


def _parse_truth(match_metadata: dict) -> dict:
    """Per-player goal / own-goal budgets, team scores and team-of map from ``match.json``."""
    home_id = _cid(match_metadata["home_team"]["id"])
    away = match_metadata.get("away_team") or {}
    away_id = _cid(away.get("id")) if away.get("id") is not None else None
    players = match_metadata.get("players") or []
    goal: dict[str, int] = {}
    own: dict[str, int] = {}
    team_of: dict[str, str] = {}
    for p in players:
        if p.get("id") is None:
            continue
        pid = _cid(p["id"])
        goal[pid] = int(p.get("goal") or 0)
        own[pid] = int(p.get("own_goal") or 0)
        if p.get("team_id") is not None:
            team_of[pid] = _cid(p["team_id"])
    return {
        "home_id": home_id,
        "away_id": away_id,
        "home_score": match_metadata.get("home_team_score"),
        "away_score": match_metadata.get("away_team_score"),
        "goal": goal,
        "own": own,
        "team_of": team_of,
    }


def _abs_scores(pp: pd.DataFrame, home_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Absolute ``(home_score, away_score)`` per possession from the possessing-team-relative GI score."""
    team = [_cid(t) for t in pp["team_id"]]
    ts = pd.to_numeric(pp["team_score"], errors="coerce").to_numpy()
    os = pd.to_numeric(pp["opponent_team_score"], errors="coerce").to_numpy()
    n = len(pp)
    home = np.full(n, np.nan)
    away = np.full(n, np.nan)
    for i in range(n):
        if same_id(team[i], home_id):
            home[i], away[i] = ts[i], os[i]
        else:
            home[i], away[i] = os[i], ts[i]
    return home, away


def _score_changes(home: np.ndarray, away: np.ndarray, home_id: str, away_id: str | None) -> list[tuple[int, str]]:
    """``(possession_index, scoring_team_id)`` for each possession where a team's score first increments."""
    changes: list[tuple[int, str]] = []
    prev_h = prev_a = 0.0
    for i in range(len(home)):
        h, a = home[i], away[i]
        if not (np.isfinite(h) and np.isfinite(a)):
            continue
        if h > prev_h:
            changes.append((i, home_id))
        elif a > prev_a and away_id is not None:
            changes.append((i, away_id))
        prev_h, prev_a = max(prev_h, h), max(prev_a, a)
    return changes


def _kickoff_anchors(
    changes: list[tuple[int, str]],
    pp_team: list[str],
    start_type: list[str],
    home_id: str,
    away_id: str | None,
) -> list[tuple[int, str]]:
    """Shift each score change to its post-goal KICK-OFF anchor (spec section 6.4).

    The post-goal kick-off is the first CONCEDING-team possession at/after the score increment (and before
    the next score change) with ``start_type == "unknown"``; the goal episode ends there. When no such
    kick-off is found, the anchor falls back to the score-change possession itself. Gating on the kick-off
    (not the raw increment) bounds the goal's candidate search to the right episode.
    """
    out: list[tuple[int, str]] = []
    for k, (i, team_t) in enumerate(changes):
        conceding = away_id if same_id(team_t, home_id) else home_id
        bound = changes[k + 1][0] if k + 1 < len(changes) else len(pp_team)
        anchor = i
        for j in range(i, bound):
            if start_type[j] == "unknown" and conceding is not None and same_id(pp_team[j], conceding):
                anchor = j
                break
        out.append((anchor, team_t))
    return out


def reconcile_goals(actions: pd.DataFrame, pp: pd.DataFrame, match_metadata: dict) -> tuple[pd.DataFrame, GoalsReport]:
    """Repair the goals against ``match.json`` (spec section 6). Returns ``(actions, GoalsReport)``.

    ``actions`` is the SkillCorner SPADL frame (native + derived restart rows, pre-finalize). ``pp`` is the
    chronological ``player_possession`` frame carrying the GI running score and interruption tags. Repaired
    rows set ``result_source="inferred"`` and ``repair_kind``; own goals add an ADR-018 ``bad_touch`` row.
    """
    truth = _parse_truth(match_metadata)
    home_id, away_id = truth["home_id"], truth["away_id"]
    goal_budget: dict[str, int] = dict(truth["goal"])
    own_budget: dict[str, int] = dict(truth["own"])
    team_of: dict[str, str] = truth["team_of"]

    pp = pp.reset_index(drop=True)
    pp_team = [_cid(t) for t in pp["team_id"]]
    pp_player = [_cid(p) if pd.notna(p) else None for p in pp["player_id"]]
    end_type = pp["end_type"].fillna("").astype(str).tolist() if "end_type" in pp.columns else [""] * len(pp)
    gi_after = (
        pp["game_interruption_after"].fillna("").astype(str).tolist()
        if "game_interruption_after" in pp.columns
        else [""] * len(pp)
    )
    gi_before = (
        pp["game_interruption_before"].fillna("").astype(str).tolist()
        if "game_interruption_before" in pp.columns
        else [""] * len(pp)
    )
    start_type = pp["start_type"].fillna("").astype(str).tolist() if "start_type" in pp.columns else [""] * len(pp)
    event_id = pp["event_id"].astype("object").tolist()

    home, away = _abs_scores(pp, home_id)
    # Score increments, then shifted to their post-goal kick-off anchor (section 6.4).
    changes = _kickoff_anchors(_score_changes(home, away, home_id, away_id), pp_team, start_type, home_id, away_id)

    # Map a possession event_id -> the action row index (native rows carry original_event_id == event_id).
    eid_to_action: dict[object, int] = {}
    for idx, eid in zip(actions.index, actions["original_event_id"], strict=True):
        if eid is not None and eid not in eid_to_action:
            eid_to_action[eid] = idx

    actions = actions.copy()
    repaired: dict[str, int] = {}
    ambiguous = 0
    unresolved = 0
    new_rows: list[dict] = []

    def _bump(cls: str) -> None:
        repaired[cls] = repaired.get(cls, 0) + 1

    def _consume_goal(pid: str) -> bool:
        if goal_budget.get(pid, 0) > 0:
            goal_budget[pid] -= 1
            return True
        return False

    def _consume_own(pid: str) -> bool:
        if own_budget.get(pid, 0) > 0:
            own_budget[pid] -= 1
            return True
        return False

    def _set_shot_success(poss_i: int, cls: str) -> None:
        aidx = eid_to_action.get(event_id[poss_i])
        if aidx is None:
            return
        actions.loc[aidx, "result_id"] = spadlconfig.result_id["success"]
        actions.loc[aidx, "result_source"] = "inferred"
        actions.loc[aidx, "repair_kind"] = cls

    def _own_goal_row(poss_i: int, conceding_team: str, og_player: str | None, cls: str) -> None:
        # ADR-018: an own goal is a bad_touch with result owngoal for the conceding own-goal player.
        new_rows.append(
            {
                "game_id": str(match_metadata.get("id", "unknown")),
                "original_event_id": event_id[poss_i],
                "action_id": 0,
                "period_id": int(pp["period_id"].iloc[poss_i]),
                "period_source": _period_source(pp, poss_i),
                "time_seconds": _own_goal_time(poss_i),
                "team_id": conceding_team,
                "player_id": og_player if og_player is not None else pd.NA,
                "start_x": float("nan"),
                "start_y": float("nan"),
                "end_x": _OWN_GOAL_END[0],
                "end_y": _OWN_GOAL_END[1],
                "type_id": spadlconfig.actiontype_id["bad_touch"],
                "result_id": spadlconfig.result_id["owngoal"],
                "result_source": "inferred",
                "bodypart_id": spadlconfig.bodypart_id["foot"],
                "action_provenance": "derived",
                "restart_evidence": pd.NA,
                "type_source": pd.NA,
                "repair_kind": cls,
            }
        )

    def _own_goal_time(poss_i: int) -> float:
        # End of the last possession in the goal episode carrying a goal interruption; else the end of
        # the last possession before the score change (spec section 6.3). Here: the possession's time.
        return float(pp["time_seconds"].iloc[poss_i])

    def _episode_bounds(change_idx: int, pos_in_changes: int) -> tuple[int, int]:
        start = changes[pos_in_changes - 1][0] + 1 if pos_in_changes > 0 else 0
        return start, change_idx

    # GI per-player goal_for attribution, and the swap set: a team where one player is over-credited and
    # another under-credited vs match.json (section 6.2). Swap is REPORTED, never row-repaired
    # (section 6.3), so its score changes skip the pass-2 repairs below (otherwise missing_goal would
    # credit the wrong teammate's shot -- a wrong assignment). A cleanly-attributed match has no swap.
    gi_credited: dict[str, int] = {}
    for i in range(len(pp)):
        pid_i = pp_player[i]
        if pid_i is not None and ((end_type[i] == "shot" and gi_after[i] == "goal_for") or gi_before[i] == "goal_for"):
            gi_credited[pid_i] = gi_credited.get(pid_i, 0) + 1
    swap_teams: set[str] = set()
    for team in {home_id} | ({away_id} if away_id is not None else set()):
        on_team = [p for p in team_of if same_id(team_of[p], team)]
        over = any(gi_credited.get(p, 0) > truth["goal"].get(p, 0) for p in on_team)
        under = any(gi_credited.get(p, 0) < truth["goal"].get(p, 0) for p in on_team)
        if over and under:
            swap_teams.add(team)
    swap = len(swap_teams)

    # --- Pass 1: explained score changes (normal / scored_tag / own_goal_inverted) ---
    handled: set[int] = set()
    for k, (idx, team_t) in enumerate(changes):
        lo, hi = _episode_bounds(idx, k)
        episode = range(lo, hi + 1)
        done = False
        # normal: a T shot possession tagged goal_for, correct scorer in the metadata.
        for i in episode:
            if same_id(pp_team[i], team_t) and end_type[i] == "shot" and gi_after[i] == "goal_for":
                pid = pp_player[i]
                if pid is not None and _consume_goal(pid):
                    _set_shot_success(i, "normal")
                    _bump("normal")
                    done = True
                    break
        if done:
            handled.add(k)
            continue
        # scored_tag: the scorer's shot carries only game_interruption_before == goal_for.
        for i in episode:
            if same_id(pp_team[i], team_t) and end_type[i] == "shot" and gi_before[i] == "goal_for":
                pid = pp_player[i]
                if pid is not None and _consume_goal(pid):
                    _set_shot_success(i, "scored_tag")
                    _bump("scored_tag")
                    done = True
                    break
        if done:
            handled.add(k)
            continue
        # own_goal_inverted: T's shot carries goal_against and the conceding kick-off carries goal_for.
        inv = any(same_id(pp_team[i], team_t) and gi_after[i] == "goal_against" for i in episode)
        conceding = _other(team_t, home_id, away_id)
        if inv and conceding is not None:
            og_pid = _unique_budget_holder(own_budget, team_of, conceding)
            if og_pid is not None and _consume_own(og_pid):
                _own_goal_row(hi, conceding, og_pid, "own_goal_inverted")
                _bump("own_goal_inverted")
                handled.add(k)

    # --- Pass 2: unexplained score changes (missing_goal / own_goal_unsignalled / swap / ambiguous) ---
    for k, (idx, team_t) in enumerate(changes):
        if k in handled:
            continue
        if team_t in swap_teams:
            continue  # swap is reported (above), never row-repaired (section 6.3)
        lo, hi = _episode_bounds(idx, k)
        episode = range(lo, hi + 1)
        conceding = _other(team_t, home_id, away_id)
        og_candidate = _unique_budget_holder(own_budget, team_of, conceding) if conceding is not None else None
        deficit = _unique_budget_holder(goal_budget, team_of, team_t)
        has_goal_for_shot = any(
            same_id(pp_team[i], team_t) and end_type[i] == "shot" and gi_after[i] == "goal_for" for i in episode
        )
        no_goal_row = not has_goal_for_shot and not any(
            same_id(pp_team[i], team_t) and gi_before[i] == "goal_for" for i in episode
        )

        # own_goal_surplus: GI credits a T shot but that scorer's metadata goals are exhausted (no T
        # deficit) and a unique opponent own-goal candidate exists -> the surplus goal is an own goal.
        if has_goal_for_shot and deficit is None and og_candidate is not None and conceding is not None:
            if _consume_own(og_candidate):
                _own_goal_row(hi, conceding, og_candidate, "own_goal_surplus")
                _bump("own_goal_surplus")
                continue

        # own_goal_unsignalled: no goal row, no T deficit, a unique opponent own-goal candidate.
        if no_goal_row and deficit is None and og_candidate is not None and conceding is not None:
            if _consume_own(og_candidate):
                _own_goal_row(hi, conceding, og_candidate, "own_goal_unsignalled")
                _bump("own_goal_unsignalled")
                continue

        # missing_goal: exactly one T player has a goal deficit and no own-goal candidate.
        if deficit is not None and og_candidate is None:
            penalty = any(gi_before[i].startswith("penalty") or gi_after[i].startswith("penalty") for i in episode)
            # reading (i): a derived restart (free kick / corner / goal kick) immediately before the goal
            # with no possession between -> retype that ONE row instead of adding a shot (spec 6.6).
            ridx = None
            if not penalty:
                t_lo, t_hi = float(pp["time_seconds"].iloc[lo]), float(pp["time_seconds"].iloc[hi])
                rmask = (
                    (actions["action_provenance"] == "derived")
                    & actions["restart_evidence"].notna()
                    & actions["time_seconds"].between(t_lo, t_hi)
                )
                cands = [i for i in actions.index[rmask] if same_id(actions.loc[i, "team_id"], team_t)]
                ridx = max(cands, key=lambda i: cast("float", actions.loc[i, "time_seconds"])) if cands else None
            last_shot = _last_shot_in_episode(pp_player, end_type, episode, deficit)
            if ridx is not None:
                cur = int(cast("int", actions.loc[ridx, "type_id"]))
                fk_ids = (spadlconfig.actiontype_id["freekick_crossed"], spadlconfig.actiontype_id["freekick_short"])
                actions.loc[ridx, "type_id"] = (
                    spadlconfig.actiontype_id["shot_freekick"] if cur in fk_ids else spadlconfig.actiontype_id["shot"]
                )
                actions.loc[ridx, "result_id"] = spadlconfig.result_id["success"]
                actions.loc[ridx, "result_source"] = "inferred"
                actions.loc[ridx, "repair_kind"] = "reading_i"
                actions.loc[ridx, "player_id"] = deficit
                _consume_goal(deficit)
                _bump("reading_i")
            elif penalty:
                # A scored penalty is ONE shot_penalty row by the deficit player (the taker) at the
                # constant spot (spec 6.5 / 9.3). Bound the shot to the PENALTY SUB-EPISODE (the penalty
                # marker onward), never the whole score-change episode -- otherwise a distant earlier miss
                # is rewritten to the spot and lost (B-05). Rewrite a native shot there if present, else
                # emit a derived shot_penalty anchored at the penalty marker.
                pen_i = next(
                    (i for i in episode if gi_before[i].startswith("penalty") or gi_after[i].startswith("penalty")),
                    lo,
                )
                pen_shot = _last_shot_in_episode(pp_player, end_type, range(pen_i, hi + 1), deficit)
                aidx = eid_to_action.get(event_id[pen_shot]) if pen_shot is not None else None
                if aidx is not None:
                    actions.loc[aidx, "type_id"] = spadlconfig.actiontype_id["shot_penalty"]
                    actions.loc[aidx, "start_x"] = _PENALTY_SPOT[0]
                    actions.loc[aidx, "start_y"] = _PENALTY_SPOT[1]
                    actions.loc[aidx, "result_id"] = spadlconfig.result_id["success"]
                    actions.loc[aidx, "result_source"] = "inferred"
                    actions.loc[aidx, "repair_kind"] = "missing_goal"
                else:
                    new_rows.append(_derived_shot_row(match_metadata, pp, pen_i, team_t, deficit, True))
                _consume_goal(deficit)
                _bump("missing_goal")
            elif last_shot is not None:
                _consume_goal(deficit)
                _set_shot_success(last_shot, "missing_goal")
                _bump("missing_goal")
            else:
                _consume_goal(deficit)
                new_rows.append(_derived_shot_row(match_metadata, pp, hi, team_t, deficit, False))
                _bump("missing_goal")
            continue

        # Otherwise reported, not repaired: ambiguous when MORE THAN ONE candidate fits (the section 6.2
        # og / og-or-swap / ambiguous_missing classes), unresolved when none does.
        n_deficit = sum(1 for p, nn in goal_budget.items() if nn > 0 and same_id(team_of.get(p, ""), team_t))
        n_og = sum(
            1
            for p, nn in own_budget.items()
            if nn > 0 and conceding is not None and same_id(team_of.get(p, ""), conceding)
        )
        if n_deficit > 1 or n_og > 1 or (deficit is not None and og_candidate is not None):
            ambiguous += 1
        else:
            unresolved += 1

    if new_rows:
        # Align appended rows to the caller's column set: pre-finalize this is the 19 internal columns;
        # a caller passing a finalized frame (extra columns such as shot_blocked) gets NA for those.
        appended = pd.DataFrame(new_rows).reindex(columns=actions.columns)
        actions = pd.concat([actions, appended], ignore_index=True)

    if swap:
        repaired["swap"] = swap

    # Coverage gap: the GI final score differs from match.json.
    gi_home = int(np.nanmax(home)) if np.isfinite(np.nanmax(home)) else 0
    gi_away = int(np.nanmax(away)) if np.isfinite(np.nanmax(away)) else 0
    coverage_gap = (truth["home_score"] is not None and truth["home_score"] != gi_home) or (
        truth["away_score"] is not None and truth["away_score"] != gi_away
    )

    report = GoalsReport(
        repaired=repaired,
        ambiguous=ambiguous,
        coverage_gap=bool(coverage_gap),
        unresolved=unresolved,
        final_home_goals=gi_home,
        final_away_goals=gi_away,
    )
    return actions, report


def _other(team: str, home_id: str, away_id: str | None) -> str | None:
    """The other team id, or ``None`` when the opponent is unresolved."""
    if away_id is None:
        return None
    return away_id if same_id(team, home_id) else home_id


def _unique_budget_holder(budget: Mapping[str, int], team_of: Mapping[str, str], team: str) -> str | None:
    """The single player on ``team`` with a remaining budget, or ``None`` when zero or more than one."""
    holders = [pid for pid, n in budget.items() if n > 0 and same_id(team_of.get(pid, ""), team)]
    return holders[0] if len(holders) == 1 else None


def _last_shot_in_episode(pp_player, end_type, episode, scorer) -> int | None:
    """The DEFICIT PLAYER's last native shot in the episode (spec section 6.3), not the team's.

    Crediting the team's last shot miscredits a penalty to the player who WON it rather than the taker,
    and leaves the real scorer (the deficit player) without a goal -- the per-player G3 defect B-05.
    """
    last = None
    for i in episode:
        if scorer is not None and same_id(pp_player[i], scorer) and end_type[i] == "shot":
            last = i
    return last


def _period_source(pp: pd.DataFrame, poss_i: int) -> str:
    """The period source of possession ``poss_i`` (TF-46 Part 2 spec section 26.5): a row whose period comes from
    that possession carries it. ``"native"`` when the possession frame has no ``period_source`` column."""
    return str(pp["period_source"].iloc[poss_i]) if "period_source" in pp.columns else "native"


def _derived_shot_row(match_metadata, pp, poss_i, team_t, scorer, penalty) -> dict:
    """A derived scoring shot for a ``missing_goal`` with no native shot (spec section 6.3)."""
    tid = spadlconfig.actiontype_id["shot_penalty"] if penalty else spadlconfig.actiontype_id["shot"]
    start = _PENALTY_SPOT if penalty else (float("nan"), float("nan"))
    return {
        "game_id": str(match_metadata.get("id", "unknown")),
        "original_event_id": pp["event_id"].astype("object").iloc[poss_i],
        "action_id": 0,
        "period_id": int(pp["period_id"].iloc[poss_i]),
        "period_source": _period_source(pp, poss_i),
        "time_seconds": float(pp["time_seconds"].iloc[poss_i]),
        "team_id": team_t,
        "player_id": scorer if scorer is not None else pd.NA,
        "start_x": start[0],
        "start_y": start[1],
        "end_x": float(spadlconfig.field_length),
        "end_y": spadlconfig.field_width / 2.0,
        "type_id": tid,
        "result_id": spadlconfig.result_id["success"],
        "result_source": "inferred",
        "bodypart_id": spadlconfig.bodypart_id["foot"],
        "action_provenance": "derived",
        "restart_evidence": pd.NA,
        "type_source": pd.NA,
        "repair_kind": "missing_goal",
    }
