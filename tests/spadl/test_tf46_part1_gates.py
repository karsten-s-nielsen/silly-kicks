"""TF-46 Arm 1 Part 1 acceptance gates on the committed opendata fixtures (Task 2.8, spec section 10.2).

Non-circular (A1-IMPL-04 / B-06): the restart census is pinned from the INDEPENDENT oracle
``scripts.measure_tf46_census.expected_restart_census`` (which re-derives counts from raw GI, never from
the converter). ``test_census_pins_match_independent_oracle`` ties the pins to that oracle;
``test_g_census`` ties the CONVERTER output to the pins. Transitively, a one-sided ``derive_restarts``
miscount fails -- the converter is never checked against itself. Offline: no token, no network.

- G-census: converter derived-restart per-type (free-kick combined) and per-restart_evidence counts match
  the oracle-pinned census.
- G2: no native row carries a restart type; the derived-restart count equals the oracle total (one row
  per restart, none duplicated or dropped).
- G3: per-player AND team goals equal ``match.json`` on all 20 opendata (0 wrong assignments).
- G4: action_id is the dense chronological range (``_assert_chronological_action_id`` ran in
  ``_finalize_output``); every derived restart time is finite and inside its period's span (the marker
  bracket, enforced by the placeholder-time construction).
- G5: no coverage gap and no unresolved goal -- the post-goal-kick-off / score-change anchor reconciled
  every goal.
"""

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pandas as pd

from scripts.measure_tf46_census import expected_restart_census
from silly_kicks.id_compat import canonical_id as _ci
from silly_kicks.spadl import config as spadlcfg
from silly_kicks.spadl import skillcorner


def _cid(value: object) -> str:
    return str(_ci(value))


_OPEN = Path(__file__).parents[1] / "datasets" / "skillcorner" / "opendata"

# Pinned from scripts/measure_tf46_census.py (the independent oracle), 2026-10-20. Free-kick combined
# (crossed+short); the crossed/short split is the section 5.7 box rule, validated by its own unit test.
_PIN_TYPE_FK_COMBINED = {"corner": 176, "freekick": 461, "goalkick": 315, "throw_in": 890}
_PIN_EVIDENCE = {"chain": 45, "chain_ambiguous": 50, "tag_only": 98, "tagged": 1576, "untagged": 73}
_PIN_RESTART_TOTAL = sum(_PIN_TYPE_FK_COMBINED.values())  # == sum(_PIN_EVIDENCE.values())

_RESTART_TYPE_IDS = {
    spadlcfg.actiontype_id[k]
    for k in ("goalkick", "throw_in", "corner_crossed", "corner_short", "freekick_crossed", "freekick_short")
}
_SHOT_IDS = {spadlcfg.actiontype_id[k] for k in ("shot", "shot_penalty", "shot_freekick")}
_ID_TO_NAME = {i: nm for i, nm in enumerate(spadlcfg.actiontypes)}
# Map the converter's SPADL type names to the oracle's base labels (free-kick + corner combined).
_NAME_TO_ORACLE = {
    "corner_crossed": "corner",
    "corner_short": "corner",
    "throw_in": "throw_in",
    "goalkick": "goalkick",
    "freekick_crossed": "freekick",
    "freekick_short": "freekick",
}


def _fixtures() -> list[tuple[str, pd.DataFrame, dict]]:
    out = []
    for csv in sorted(_OPEN.glob("*_events_slim.csv")):
        mid = csv.name.split("_")[0]
        meta = json.loads((_OPEN / f"{mid}_match.json").read_text(encoding="utf-8"))
        out.append((mid, pd.read_csv(csv, low_memory=False), meta))
    return out


@lru_cache(maxsize=1)
def _converted() -> tuple:
    return tuple((mid, skillcorner.convert_to_actions(ev, meta)[0], meta) for mid, ev, meta in _fixtures())


def _converter_restart_counts() -> tuple[Counter, Counter]:
    type_fk_combined: Counter = Counter()
    evidence: Counter = Counter()
    for _mid, actions, _meta in _converted():
        restarts = actions[(actions["action_provenance"] == "derived") & actions["restart_evidence"].notna()]
        for tid in restarts["type_id"]:
            name = _ID_TO_NAME.get(int(tid), "unknown")
            type_fk_combined[_NAME_TO_ORACLE.get(name, name)] += 1
        for ev in restarts["restart_evidence"]:
            evidence[str(ev)] += 1
    return type_fk_combined, evidence


def test_census_pins_match_independent_oracle():
    # Ties the pinned constants to the INDEPENDENT oracle (raw-GI re-derivation, no converter).
    per_type: Counter = Counter()
    per_evidence: Counter = Counter()
    for _mid, events, meta in _fixtures():
        t, e = expected_restart_census(events, meta)
        per_type.update(t)
        per_evidence.update(e)
    assert dict(per_type) == _PIN_TYPE_FK_COMBINED
    assert dict(per_evidence) == _PIN_EVIDENCE


def test_g_census_converter_matches_pins():
    type_fk_combined, evidence = _converter_restart_counts()
    assert dict(type_fk_combined) == _PIN_TYPE_FK_COMBINED
    assert dict(evidence) == _PIN_EVIDENCE


def test_g2_no_native_restart_and_one_row_per_restart():
    total_derived = 0
    for mid, actions, _meta in _converted():
        native_restart = actions[
            (actions["action_provenance"] == "native") & actions["type_id"].isin(_RESTART_TYPE_IDS)
        ]
        assert native_restart.empty, f"{mid}: {len(native_restart)} native rows carry a restart type"
        total_derived += int(((actions["action_provenance"] == "derived") & actions["restart_evidence"].notna()).sum())
    assert total_derived == _PIN_RESTART_TOTAL  # one derived row per restart; none dropped or duplicated


def test_g3_per_player_and_team_goals_equal_match_json():
    for mid, actions, meta in _converted():
        credit: Counter = Counter()
        own: Counter = Counter()
        for tid, rid, team, player in zip(
            actions["type_id"], actions["result_id"], actions["team_id"], actions["player_id"], strict=True
        ):
            if tid in _SHOT_IDS and rid == spadlcfg.result_id["success"]:
                credit[_cid(player)] += 1
            if rid == spadlcfg.result_id["owngoal"]:
                own[_cid(team)] += 1
        # per-player goal credit equals match.json
        mj_player = {_cid(p["id"]): int(p.get("goal") or 0) for p in meta["players"]}
        for pid in set(credit) | set(mj_player):
            assert credit.get(pid, 0) == mj_player.get(pid, 0), f"{mid}: player {pid} credit mismatch"
        # team scorelines (goals + opponent own goals) equal match.json
        home_id, away_id = _cid(meta["home_team"]["id"]), _cid(meta["away_team"]["id"])
        team_goals = Counter()
        for pid, c in credit.items():
            team_goals[mj_player_team(meta, pid)] += c
        sp_home = team_goals[home_id] + own[away_id]
        sp_away = team_goals[away_id] + own[home_id]
        assert sp_home == meta.get("home_team_score") and sp_away == meta.get("away_team_score"), mid


def mj_player_team(meta: dict, pid: str) -> str:
    for p in meta["players"]:
        if _cid(p["id"]) == pid:
            return _cid(p["team_id"])
    return pid


def test_g4_chronology_and_marker_bracket():
    for mid, actions, _meta in _converted():
        assert list(actions["action_id"]) == list(range(len(actions))), f"{mid}: action_id not 0..n-1"
        derived = actions[(actions["action_provenance"] == "derived") & actions["restart_evidence"].notna()]
        assert derived["time_seconds"].notna().all(), f"{mid}: a derived restart has a NaN time"
        for period, grp in actions.groupby("period_id"):
            native = grp[grp["action_provenance"] == "native"]
            if native.empty:
                continue
            lo, hi = native["time_seconds"].min(), native["time_seconds"].max()
            d = derived[derived["period_id"] == period]["time_seconds"]
            assert ((d >= lo - 1.0) & (d <= hi + 1.0)).all(), f"{mid} p{period}: derived time outside bracket"


def test_g5_anchor_no_coverage_gap():
    for mid, _ev, meta in _fixtures():
        _, report = skillcorner.convert_to_actions(
            pd.read_csv(_OPEN / f"{mid}_events_slim.csv", low_memory=False), meta
        )
        assert report.goals_coverage_gap is False, f"{mid}: coverage gap (anchor did not reconcile)"
        assert report.goals_unresolved == 0 and report.goals_ambiguous == 0, mid


def _convert_named(mid: str):
    meta = json.loads((_OPEN / f"{mid}_match.json").read_text(encoding="utf-8"))
    return skillcorner.convert_to_actions(pd.read_csv(_OPEN / f"{mid}_events_slim.csv", low_memory=False), meta)


def test_named_goal_cases_hit_expected_repair_class():
    # The two section-10.1 GOAL cases, pinned individually to their repair class (A1-IMPL-06).
    actions_oi, report_oi = _convert_named("2016236")  # own-goal kick-off
    assert report_oi.goals_repaired.get("own_goal_inverted", 0) >= 1
    assert (actions_oi["result_id"] == spadlcfg.result_id["owngoal"]).sum() >= 1

    actions_pen, report_pen = _convert_named("2006229")  # penalty kick-off
    assert report_pen.goals_repaired.get("missing_goal", 0) >= 1
    pen = actions_pen[actions_pen["type_id"] == spadlcfg.actiontype_id["shot_penalty"]]
    assert len(pen) >= 1 and (pen["start_x"] == 94.0).all() and (pen["start_y"] == 34.0).all()


def test_named_wrong_throw_in_for_case_2011166():
    # section-10.1 RESTART case: the wrong `throw_in_for` tag is handled -- throw-in restarts derive and
    # the match reconciles cleanly (no unresolved / ambiguous / coverage gap, no integrity raise).
    actions, report = _convert_named("2011166")
    throw_in = actions[
        (actions["action_provenance"] == "derived")
        & actions["restart_evidence"].notna()
        & (actions["type_id"] == spadlcfg.actiontype_id["throw_in"])
    ]
    assert len(throw_in) == 49  # pinned (independent census route in measure_tf46_census.py)
    assert report.goals_unresolved == 0 and report.goals_ambiguous == 0 and report.goals_coverage_gap is False
