"""Independent TF-46 restart + goals census on the committed opendata fixtures (Task 2.8).

INDEPENDENT ORACLE (A1-IMPL-04 / TF46C-PLAN-06): this script must NOT import ``convert_to_actions`` /
``derive_restarts`` / ``reconcile_goals``. It re-derives the expected restart census straight from the raw
GI markers/tags with a SEPARATE implementation of the spec section 5.1 rules, and the goal census from
``match.json``. The gate (``tests/spadl/test_tf46_part1_gates.py``) runs the converter and asserts its
output against the constants pinned from THIS oracle -- so a one-sided ``derive_restarts`` miscount fails
the gate instead of passing against itself. Only the pure coordinate/time helpers are reused
(``_parse_time_start`` / ``_to_period_relative`` / ``sort_actions_chronologically``), never the derivation.

Per-type counts combine the free-kick subtype (``freekick`` = crossed + short): the crossed/short split is
the box rule, validated separately by the section 5.7 unit test, not by this derivation-count oracle.

Offline: reads the committed slim fixtures, no token, no network.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from silly_kicks.id_compat import canonical_id, canonical_id_series, same_id
from silly_kicks.spadl.skillcorner import _parse_time_start, _to_period_relative, sort_actions_chronologically

_RESTART_BASES = {"corner", "free_kick", "goal_kick", "throw_in"}
_TYPE_LABEL = {"corner": "corner", "throw_in": "throw_in", "goal_kick": "goalkick", "free_kick": "freekick"}


def _cid(value: object) -> str:
    """Canonical id as a plain ``str`` (ids here are non-null by construction)."""
    return str(canonical_id(value))


def _parse(value: str) -> tuple[str | None, str | None]:
    if value.endswith("_for"):
        base, side = value[:-4], "for"
    elif value.endswith("_against"):
        base, side = value[:-8], "against"
    else:
        return None, None
    return (base, side) if base in _RESTART_BASES else (None, None)


def expected_restart_census(events: pd.DataFrame, meta: dict) -> tuple[Counter, Counter]:
    """``(per_type, per_evidence)`` expected restart counts, re-derived independently from raw GI."""
    pp = events[events["event_type"] == "player_possession"].copy()
    pp = pp[pp["player_id"].notna() & pp["team_id"].notna()].copy()
    pp["team_id"] = pp["team_id"].astype(str)
    pp["time_seconds"] = _to_period_relative(_parse_time_start(pp["time_start"]), pp["period"])
    pp = sort_actions_chronologically(pp, by=("period", "time_seconds")).reset_index(drop=True)

    home = _cid(meta["home_team"]["id"])
    teams = {_cid(t) for t in pp["team_id"].unique()}
    others = teams - {home}
    opp = next(iter(others)) if others else None
    team = [_cid(t) for t in pp["team_id"]]
    gi_after = pp["game_interruption_after"].fillna("").astype(str).tolist()
    gi_before = pp["game_interruption_before"].fillna("").astype(str).tolist()
    period = pp["period"].to_numpy()
    n = len(pp)

    def other(t: str) -> str | None:
        return None if opp is None else (home if same_id(t, opp) else opp)

    tag = [_parse(g) for g in gi_before]
    tag_restarter = [(team[i] if tag[i][1] == "for" else other(team[i])) if tag[i][0] else None for i in range(n)]

    per_type: Counter = Counter()
    per_evidence: Counter = Counter()

    # (a) interruption markers on the last possession before each stoppage.
    for i in range(n):
        base, side = _parse(gi_after[i])
        last_in_period = i == n - 1 or period[i] != period[i + 1]
        if base is None or last_in_period:
            continue
        rest = team[i] if side == "for" else other(team[i])
        if rest is None:
            continue
        nxt = i + 1
        normal = (
            nxt < n and tag[nxt][0] == base and tag_restarter[nxt] is not None and same_id(tag_restarter[nxt], rest)
        )
        chained = nxt < n and tag[nxt][0] is not None and not normal
        if chained:
            second_base = tag[nxt][0]
            second_rest = tag_restarter[nxt]
            to_t = second_rest is not None and same_id(second_rest, rest)
            if second_base == "free_kick":
                per_evidence["chain_ambiguous" if to_t else "chain"] += 1
            else:
                per_evidence["chain" if to_t else "chain_ambiguous"] += 1
        else:
            per_evidence["tagged" if normal else "untagged"] += 1
        per_type[_TYPE_LABEL[base]] += 1

    # (b) first-possession tags the previous marker does not explain.
    for i in range(n):
        base_i = tag[i][0]
        if base_i is None:
            continue
        if i == 0 or period[i] != period[i - 1]:
            continue
        prev = i - 1
        pbase, pside = _parse(gi_after[prev])
        prev_last = period[prev] != period[prev + 1]
        if pbase is not None and not prev_last:
            prest = team[prev] if pside == "for" else other(team[prev])
            normal = (
                tag[i][0] == pbase
                and prest is not None
                and tag_restarter[i] is not None
                and same_id(tag_restarter[i], prest)
            )
            if normal:
                continue  # explained: normal receiver of the previous marker
        if tag_restarter[i] is None:
            continue
        per_evidence["tag_only"] += 1
        per_type[_TYPE_LABEL[base_i]] += 1

    return per_type, per_evidence


_SPORTEC_KEEPER_GK = frozenset({"save", "claim", "punch", "pickUp", "throwOut", "punt"})


def _sportec_survives_as(event_type: str, gk_action: str, defensive_clearance: str) -> str | None:
    """Independent SPADL-survival classifier for a DFL event (Sportec F4 oracle).

    Returns a coarse SPADL class (``tackle`` / ``keeper`` / ``pass`` / ``setpiece`` / ``shot`` / ``foul`` /
    ``clearance``) for an event that produces a surviving SPADL action, or ``None`` for one that maps to
    ``non_action`` (dropped) or is excluded. This re-derives the type dispatch separately from
    ``sportec._build_raw_actions`` -- never imported here -- so the F4 census ties to the raw events, not to
    the converter. Only the subset of event types that can be a BallClaiming partner needs to be exact.
    """
    if event_type == "TacklingGame":
        return "tackle"
    if event_type == "Play":
        if gk_action in _SPORTEC_KEEPER_GK:
            return "keeper"
        return "pass" if gk_action == "" else None  # unrecognized GK qualifier -> non_action
    if event_type in ("FreeKick", "Corner", "ThrowIn", "GoalKick"):
        return "setpiece"
    if event_type == "ShotAtGoal":
        return "shot"
    if event_type == "Foul":
        return "foul"
    if event_type == "OtherBallAction":
        return "clearance" if defensive_clearance == "true" else None
    return None


def expected_sportec_f4_f7_census(events: pd.DataFrame) -> dict:
    """``BallClaiming`` (F4) and free-kick-crossed (F7) expected counts, re-derived from raw DFL events.

    INDEPENDENT ORACLE (TF46C Part-1 Commit 3): does NOT import ``sportec.convert_to_actions``. It
    classifies each event's SPADL survival with ``_sportec_survives_as`` and resolves every ``BallClaiming``
    to its nearest same-player surviving partner within +-0.1 s (same partner rule the converter uses),
    then counts: a derived ``interception`` for each ``InterceptedBall`` AND each ``BallClaimed`` whose
    partner is not already a ``tackle`` / ``keeper`` action (spec section 7.1 amended -- BallClaimed is an
    outfield ball win, mapped like InterceptedBall); a derived ``keeper_pick_up`` for each ``BallHeld`` with
    no surviving keeper partner (the duplicate-keeper guard); and the ``Block`` claims whose partner did not
    survive (``ball_claim_unattached`` -- now Block-only). F7 counts free kicks by ``play_penalty_box``.
    """
    ev = events.copy()
    gk = (
        ev["play_goal_keeper_action"].fillna("").astype(str)
        if "play_goal_keeper_action" in ev.columns
        else pd.Series([""] * len(ev), index=ev.index)
    )
    dc = (
        ev["otherball_defensive_clearance"].fillna("").astype(str).str.lower()
        if "otherball_defensive_clearance" in ev.columns
        else pd.Series([""] * len(ev), index=ev.index)
    )
    et = ev["event_type"].astype(str)
    cls = [_sportec_survives_as(et.iloc[k], gk.iloc[k], dc.iloc[k]) for k in range(len(ev))]
    ev["_cls"] = pd.Series(cls, index=ev.index, dtype=object)
    ev["_pid"] = canonical_id_series(ev["player_id"])  # ADR-019: canonical key, not astype(str)
    surv = ev[ev["_cls"].notna()]

    claims: Counter = Counter()
    interceptions = keeper_pick_ups = unattached = 0
    bc = ev[ev["event_type"] == "BallClaiming"]
    for _, r in bc.iterrows():
        ct = str(r["claim_type"])
        claims[ct] += 1
        pid = canonical_id(r["player_id"])  # ADR-019: canonical key, not str()
        per = int(r["period"])
        t = float(r["timestamp_seconds"])
        cand = surv[(surv["_pid"] == pid) & (surv["period"] == per) & ((surv["timestamp_seconds"] - t).abs() <= 0.1)]
        has_partner = len(cand) > 0
        if has_partner:
            j = (cand["timestamp_seconds"] - t).abs().idxmin()
            partner_cls = cand.loc[j, "_cls"]
            near_keeper = bool((cand["_cls"] == "keeper").any())
        else:
            partner_cls = None
            near_keeper = False
        if ct in ("InterceptedBall", "BallClaimed"):
            # Both map to a derived interception unless the partner is already a tackle/keeper touch
            # (spec section 7.1 amended: BallClaimed follows the InterceptedBall rule).
            if not (has_partner and partner_cls in ("tackle", "keeper")):
                interceptions += 1
        elif ct == "BallHeld":
            if not near_keeper:
                keeper_pick_ups += 1
        elif not has_partner:  # Block with no surviving partner -- the only unattached class
            unattached += 1

    fk = ev[ev["event_type"] == "FreeKick"]
    pb = (
        fk["play_penalty_box"].astype(str).str.lower() if "play_penalty_box" in fk.columns else pd.Series([], dtype=str)
    )
    return {
        "ball_claim_counts": dict(sorted(claims.items())),
        "interceptions_added": interceptions,
        "keeper_pick_ups_added": keeper_pick_ups,
        "ball_claim_unattached": unattached,
        "freekick_crossed": int((pb == "true").sum()),
        "freekick_short": int((~(pb == "true")).sum()),
        "freekick_inferred": int((~pb.isin(["true", "false"])).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Independent TF-46 restart + goals census (opendata).")
    ap.add_argument(
        "--opendata-dir",
        default="tests/datasets/skillcorner/opendata",
        help="Directory of the committed <id>_events_slim.csv + <id>_match.json fixtures.",
    )
    ap.add_argument(
        "--idsse-dir",
        default="tests/datasets/idsse",
        help="Directory of the committed <bare>_events_native.parquet IDSSE fixtures (Sportec F4/F7 census).",
    )
    args = ap.parse_args()
    root = Path(args.opendata_dir)

    per_type: Counter = Counter()
    per_evidence: Counter = Counter()
    n = 0
    goal_total = 0
    for csv in sorted(root.glob("*_events_slim.csv")):
        mid = csv.name.split("_")[0]
        meta = json.loads((root / f"{mid}_match.json").read_text(encoding="utf-8"))
        events = pd.read_csv(csv, low_memory=False)
        t, e = expected_restart_census(events, meta)
        per_type.update(t)
        per_evidence.update(e)
        goal_total += int(meta.get("home_team_score") or 0) + int(meta.get("away_team_score") or 0)
        n += 1

    census = {
        "n_matches": n,
        "restart_type_counts_fk_combined": dict(sorted(per_type.items())),
        "restart_evidence_counts": dict(sorted(per_evidence.items())),
        "match_json_goal_total": goal_total,
    }

    # Sportec F4 (BallClaiming) + F7 (free-kick crossed) census over the committed IDSSE fixtures.
    idsse = Path(args.idsse_dir)
    f4_claims: Counter = Counter()
    f4 = {"interceptions_added": 0, "keeper_pick_ups_added": 0, "ball_claim_unattached": 0}
    f7 = {"freekick_crossed": 0, "freekick_short": 0, "freekick_inferred": 0}
    n_idsse = 0
    for pq in sorted(idsse.glob("*_events_native.parquet")):
        c = expected_sportec_f4_f7_census(pd.read_parquet(pq))
        f4_claims.update(c["ball_claim_counts"])
        for k in f4:
            f4[k] += c[k]
        for k in f7:
            f7[k] += c[k]
        n_idsse += 1
    census["sportec_f4_f7"] = {
        "n_matches": n_idsse,
        "ball_claim_counts": dict(sorted(f4_claims.items())),
        **f4,
        **f7,
    }
    print(json.dumps(census, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
