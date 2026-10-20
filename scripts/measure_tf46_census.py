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

from silly_kicks.id_compat import canonical_id, same_id
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Independent TF-46 restart + goals census (opendata).")
    ap.add_argument(
        "--opendata-dir",
        default="tests/datasets/skillcorner/opendata",
        help="Directory of the committed <id>_events_slim.csv + <id>_match.json fixtures.",
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
    print(json.dumps(census, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
