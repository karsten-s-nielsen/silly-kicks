"""TF-46 Arm 1 Part 2 §25 Guards 1 / 1b / 2 (events-only, committed fixtures).

Guard 1  — goal reconciliation: SPADL goals per team equal the provider's final score, on every provider
            with a committed real match fixture and a resolvable final score.
Guard 1b — kick-off reconciliation (decision D3): on providers that code kick-offs (DFL, Metrica), every
            non-period-start kick-off by team T is preceded by exactly one goal for T's opponent since the
            previous kick-off, and every goal is followed by an opponent kick-off (or the period end) with
            no on-ball event in between.
Guard 2  — the exact Sportec unrecognized-event-type allow-list (empty after the §25 moves).

Per-file classification (decision D2 / DL ruling): every committed match-EVENT fixture file is classified
(real_full_match / real_excerpt / slice / synthetic / stored_artifact / covered_by_g3). The derivation test
asserts the registry is total over the event-fixture domain, so a new event fixture fails until classified
(ADR-056). Non-event files (tracking frames, stored SPADL, goldens, generator scripts, docs) are outside the
goal-reconciliation domain and are not classified here.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl import sportec
from silly_kicks.spadl.schema import REPAIR_KIND_VALUES

_DATASETS = Path(__file__).parent.parent / "datasets"
_IDSSE = _DATASETS / "idsse"
IDSSE_MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]

_GOAL_TYPES = frozenset(
    {
        spadlconfig.actiontype_id["shot"],
        spadlconfig.actiontype_id["shot_freekick"],
        spadlconfig.actiontype_id["shot_penalty"],
    }
)
_SUCCESS = spadlconfig.result_id["success"]
_OWNGOAL = spadlconfig.result_id["owngoal"]


def _convert_idsse(match: str):
    meta = json.loads((_IDSSE / f"{match}_meta.json").read_text(encoding="utf-8"))
    events = pd.read_parquet(_IDSSE / f"{match}_events_native.parquet")
    return sportec.convert_to_actions(
        events,
        home_team_id="home",
        home_team_start_left=meta["home_team_start_left"],
        home_team_start_left_extratime=meta.get("home_team_start_left_extratime"),
    )


# ===========================================================================
# Per-file event-fixture classification (ADR-056 derivation)
# ===========================================================================

_CLASSES = frozenset(
    {"real_full_match", "real_excerpt", "slice", "synthetic", "stored_artifact", "covered_by_g3", "spliced_excerpt"}
)

# The event-fixture domain: committed files that are (candidates for) a SPADL converter's event input. Each
# maps to its class. Non-event files (tracking/, spadl/, goldens, scripts, docs, skillcorner unit fixtures)
# are out of this domain by the globs in _event_fixture_domain().
EVENT_FIXTURES: dict[str, str] = {
    # real full matches (sportec/IDSSE native) — Guard 1 (labeled) + Guard 1b
    **{f"idsse/{m}_events_native.parquet": "real_full_match" for m in IDSSE_MATCHES},
    "idsse/sample_match.parquet": "real_excerpt",  # a 308-event stratified sample, no final-score field
    "idsse/per_period_match.parquet": "real_full_match",
    # Metrica Sample Game 1: Guard-1 exempt (no final-score field); Guard-1b dropped by owner ruling.
    "metrica/per_period_match.parquet": "real_full_match",
    "metrica/sample_match.parquet": "real_excerpt",  # Metrica Sample Game 2, 300-event subset
    # kloppy/sportec_events.xml is a SPLICED excerpt (DL ruling 2026-10-21): 27 events of DFL-MAT-003BN1 plus
    # one FOREIGN top-level OwnGoal (EventId 182094, CurrentResult 3:3). Not one match; sportec_meta Result
    # "1:0" does not describe it. Guard 1 / 1b do NOT apply; a two-path parse-contract test covers it instead.
    "kloppy/sportec_events.xml": "spliced_excerpt",
    # Metrica Sample Game 2: Guard-1 exempt (no final-score field, mismatched meta); Guard-1b dropped (owner ruling).
    "kloppy/metrica_events.json": "real_full_match",
    "statsbomb/raw/events/7298.json": "real_full_match",
    "statsbomb/raw/events/7584.json": "real_full_match",
    "statsbomb/raw/events/3754058.json": "real_full_match",
    # synthetic
    "statsbomb/raw/events/synthetic_gk_sub.json": "synthetic",
    "statsbomb/raw/events/synthetic_gk_sub_halftime.json": "synthetic",
    "statsbomb/k2_synthetic.json": "synthetic",
    "gradientsports/synthetic_match.json": "synthetic",
    # slices
    "sportec/idsse_slice/events.xml": "slice",
    "statsbomb/three-sixty/events.json": "slice",
    # stored artifact
    "statsbomb/spadl-WorldCup-2018.h5": "stored_artifact",
    # covered by the Part-1 G3 reconciliation (the opendata slim-event files are added below)
}
EVENT_FIXTURES.update(
    {
        p.relative_to(_DATASETS).as_posix(): "covered_by_g3"
        for p in (_DATASETS / "skillcorner" / "opendata").glob("*_events_slim.csv")
    }
)

# Per real / g3 / spliced match fixture: how Guard 1 treats it — a gated METHOD or an "exempt: <reason>"
# with a recorded authority (C-6: the registry must gate, not just classify).
_GUARD1_GATED_METHODS = frozenset({"idsse_labeled", "statsbomb_per_team", "g3"})
GUARD1_DECISION: dict[str, str] = {
    **{f"idsse/{m}_events_native.parquet": "idsse_labeled" for m in IDSSE_MATCHES},
    "statsbomb/raw/events/7298.json": "statsbomb_per_team",
    "statsbomb/raw/events/7584.json": "statsbomb_per_team",
    "statsbomb/raw/events/3754058.json": "statsbomb_per_team",
    "idsse/sample_match.parquet": "exempt: 308-event stratified sample, no final-score field",
    "idsse/per_period_match.parquet": "exempt: duplicate of J03WMX (gated via idsse/J03WMX_events_native.parquet)",
    "metrica/per_period_match.parquet": "exempt: no final-score field (Metrica Sample Game 1)",
    "metrica/sample_match.parquet": "exempt: 300-event subset, no final-score field (Metrica Sample Game 2)",
    "kloppy/metrica_events.json": "exempt: no final-score field (Metrica Sample Game 2, mismatched metadata)",
    "kloppy/sportec_events.xml": "exempt: spliced excerpt; the two-path parse-contract test covers it",
}
GUARD1_DECISION.update(
    {
        p.relative_to(_DATASETS).as_posix(): "g3"
        for p in (_DATASETS / "skillcorner" / "opendata").glob("*_events_slim.csv")
    }
)

_EVENT_FIXTURE_GLOBS = (
    "idsse/*_events_native.parquet",
    "idsse/sample_match.parquet",
    "idsse/per_period_match.parquet",
    "metrica/per_period_match.parquet",
    "metrica/sample_match.parquet",
    "kloppy/sportec_events.xml",
    "kloppy/metrica_events.json",
    "statsbomb/raw/events/*.json",
    "statsbomb/k2_synthetic.json",
    "statsbomb/three-sixty/events.json",
    "statsbomb/spadl-WorldCup-2018.h5",
    "sportec/idsse_slice/events.xml",
    "gradientsports/synthetic_match.json",
    "skillcorner/opendata/*_events_slim.csv",
)


def _event_fixture_domain() -> set[str]:
    found: set[str] = set()
    for glob in _EVENT_FIXTURE_GLOBS:
        found |= {p.relative_to(_DATASETS).as_posix() for p in _DATASETS.glob(glob)}
    return found


def test_event_fixture_registry_is_total_over_the_domain():
    # ADR-056: the registry population is derived from the filesystem and asserted exactly; a new event
    # fixture in a known provider dir fails until classified.
    assert set(EVENT_FIXTURES) == _event_fixture_domain()
    assert set(EVENT_FIXTURES.values()) <= _CLASSES
    for rel in EVENT_FIXTURES:
        assert (_DATASETS / rel).exists(), rel


def test_every_real_match_fixture_has_a_guard1_decision():
    # C-6: the registry must GATE, not just classify. Every real / excerpt / g3 / spliced fixture carries a
    # Guard-1 decision — a gated method or an "exempt: <reason>" authority — and the gated sets match the
    # live Guard 1 tests.
    needs_decision = {
        p
        for p, k in EVENT_FIXTURES.items()
        if k in {"real_full_match", "real_excerpt", "covered_by_g3", "spliced_excerpt"}
    }
    assert needs_decision == set(GUARD1_DECISION)
    for path, decision in GUARD1_DECISION.items():
        assert decision in _GUARD1_GATED_METHODS or decision.startswith("exempt:"), (path, decision)
    idsse_gated = {p for p, d in GUARD1_DECISION.items() if d == "idsse_labeled"}
    assert idsse_gated == {f"idsse/{m}_events_native.parquet" for m in IDSSE_MATCHES}
    statsbomb_gated = {p for p, d in GUARD1_DECISION.items() if d == "statsbomb_per_team"}
    assert statsbomb_gated == {f"statsbomb/raw/events/{m}.json" for m in ("7298", "7584", "3754058")}


# ===========================================================================
# Guard 2 — exact unrecognized-type allow-list (§25.4)
# ===========================================================================

SPORTEC_UNRECOGNIZED_ALLOWLIST: frozenset[str] = frozenset()  # empty after the moves; a new type must be listed


def test_idsse_unrecognized_types_equal_allowlist():
    seen: set[str] = set()
    for match in IDSSE_MATCHES:
        _, report = _convert_idsse(match)
        seen |= set(report.unrecognized_counts)
    assert seen == SPORTEC_UNRECOGNIZED_ALLOWLIST  # both directions


# ===========================================================================
# Guard 1 — goal reconciliation (§25.5)
# ===========================================================================


def _idsse_final_score(match: str) -> tuple[int, int]:
    events = pd.read_parquet(_IDSSE / f"{match}_events_native.parquet")
    fw = events[events["whistle_final_result"].notna()].sort_values(["period", "timestamp_seconds"])
    home, away = (int(x) for x in str(fw.iloc[-1]["whistle_final_result"]).split(":"))
    return home, away


def _spadl_goals_by_team(actions: pd.DataFrame) -> dict[object, int]:
    """SPADL goals credited per team: successful shots to the scorer, owngoals to the opponent.

    The input is SPADL actions, where ``team_id`` carries exactly the two match teams (unresolved actors are
    NaN by the null-actor convention and are dropped; no ``"unknown"`` sentinel). C1-IMPL-10: resolve the
    own-goal opponent loud — a team set that is not exactly two makes the credit ambiguous, so fail here
    rather than silently undercount (which would surface only as a confusing Guard-1 equality failure).
    """
    teams = [t for t in actions["team_id"].dropna().unique()]
    assert len(teams) == 2, f"expected exactly two match teams in SPADL team_id, got {teams}"
    goals = {t: 0 for t in teams}
    goal_rows = actions[(actions["type_id"].isin(_GOAL_TYPES)) & (actions["result_id"] == _SUCCESS)]
    for t in teams:
        goals[t] += int((goal_rows["team_id"] == t).sum())
    owngoals = actions[actions["result_id"] == _OWNGOAL]
    for _, row in owngoals.iterrows():
        opp = [t for t in teams if t != row["team_id"]]
        assert len(opp) == 1, f"own-goal scorer team {row['team_id']!r} not one of the two match teams {teams}"
        goals[opp[0]] += 1
    return goals


def test_guard1_idsse_goals_equal_final_score():
    for match in IDSSE_MATCHES:
        actions, _ = _convert_idsse(match)
        home_goals, away_goals = _idsse_final_score(match)
        goals = _spadl_goals_by_team(actions)
        assert (goals.get("home", 0), goals.get("away", 0)) == (home_goals, away_goals), match


def test_guard1_statsbomb_goals_equal_scores():
    from tests.invariants import _loaders

    scores = json.loads((_DATASETS / "statsbomb" / "scores.json").read_text(encoding="utf-8"))["matches"]
    for match_id, score in scores.items():
        actions, _ = _loaders.load_statsbomb(int(match_id))
        goals = _spadl_goals_by_team(actions)
        # Per-team (C1-IMPL-04): each team's SPADL goals equal its recorded score; the team ids are ints.
        expected = {int(tid): n for tid, n in score["team_goals"].items()}
        observed = {int(str(t)): int(n) for t, n in goals.items()}  # keys are object-typed team ids
        assert observed == expected, match_id


# kloppy/sportec_events.xml is a SPLICED excerpt (DL ruling 2026-10-21): 27 events of DFL-MAT-003BN1 plus one
# FOREIGN top-level OwnGoal event (EventId 182094, CurrentResult 3:3, Player DFL-OBJ-0028QK, Team
# DFL-CLU-000004, TeamOwnGoalCountsFor DFL-CLU-00000A). Not one match, and sportec_meta Result "1:0" does not
# describe it, so Guard 1 / 1b do not apply. The kloppy gateway converts it correctly (a success shot + an
# own goal); the NATIVE DFL path had NO own-goal handling (the §25.2 gap decision 3 fixes). The two-path
# parse-contract test below asserts BOTH paths agree on this file.


def test_kloppy_sportec_spliced_excerpt_two_path_parse_contract():
    # DL ruling: both the kloppy gateway AND the native DFL parse + spadl.sportec must produce, for this
    # spliced excerpt, one success shot (the Freiburg goal) and one bad_touch / owngoal for the own-goal
    # scorer DFL-OBJ-0028QK. The native path FAILED before the decision-3 fix (OwnGoal was dropped). Team is
    # remapped to home/away by the native port vs the gateway's raw CLU id (an ADR-028 / ADR-001 convention),
    # so only player_id / type / result are asserted identical across the two paths, not the team string.
    pytest.importorskip("kloppy")
    from kloppy import sportec as kloppy_sportec  # type: ignore[import-not-found]
    from kloppy.domain import Provider  # type: ignore[import-not-found]
    from packaging import version

    import silly_kicks.providers.sportec.parse as parse
    from silly_kicks.spadl import kloppy as kloppy_mod

    bad_touch = spadlconfig.actiontype_id["bad_touch"]
    kloppy_dir = _DATASETS / "kloppy"

    def _assert_shot_and_own_goal(actions: pd.DataFrame) -> None:
        own = actions[(actions["type_id"] == bad_touch) & (actions["result_id"] == _OWNGOAL)]
        assert len(own) == 1
        assert str(own.iloc[0]["player_id"]) == "DFL-OBJ-0028QK"
        shots = actions[(actions["type_id"].isin(_GOAL_TYPES)) & (actions["result_id"] == _SUCCESS)]
        assert len(shots) == 1

    # gateway path
    ds = kloppy_sportec.load_event(
        event_data=str(kloppy_dir / "sportec_events.xml"), meta_data=str(kloppy_dir / "sportec_meta.xml")
    )
    original = dict(kloppy_mod._SUPPORTED_PROVIDERS)
    kloppy_mod._SUPPORTED_PROVIDERS[Provider.SPORTEC] = version.parse("3.15.0")
    try:
        gateway_actions, _ = kloppy_mod.convert_to_actions(ds, game_id="spliced_excerpt")
    finally:
        kloppy_mod._SUPPORTED_PROVIDERS.clear()
        kloppy_mod._SUPPORTED_PROVIDERS.update(original)
    _assert_shot_and_own_goal(gateway_actions)

    # native DFL parse path (match id "003BN1": the port's IDSSE id pattern; orientation is irrelevant here)
    match_info = parse.parse_dfl_match_info(str(kloppy_dir / "sportec_meta.xml"))
    bronze = parse.parse_dfl_events(str(kloppy_dir / "sportec_events.xml"), match_info=match_info, match_id="003BN1")
    native = parse.shape_events_to_native(bronze)
    # Decision 3 (f): the parse map now carries the OwnGoal field. The own goal counts for Freiburg (home).
    assert str(native.loc[native["event_type"] == "OwnGoal", "own_goal_team_own_goal_counts_for"].iloc[0]) == (
        "DFL-CLU-00000A"
    )
    native_actions, _ = sportec.convert_to_actions(native, home_team_id="home", home_team_start_left=True)
    _assert_shot_and_own_goal(native_actions)


# ===========================================================================
# Guard 1b — kick-off reconciliation (decision D3) — IDSSE only
# ===========================================================================


def _opponent_of(team: object, teams: list[object]) -> object | None:
    others = [t for t in teams if t != team]
    return others[0] if len(others) == 1 else None


def _check_kickoff_reconciliation(signals: list[dict], teams: list[object]) -> None:
    """Assert the native kick-off / goal / on-ball stream reconciles (decision D3).

    ``signals``: per native event, ``{"period", "time", "kind" in {kickoff, goal, on_ball}, "team"}`` where
    team is the kick-off's kicker or the goal's beneficiary.
    """
    for period in sorted({s["period"] for s in signals}):
        rows = sorted((s for s in signals if s["period"] == period), key=lambda s: s["time"])
        kickoffs = [s for s in rows if s["kind"] == "kickoff"]
        goals = [s for s in rows if s["kind"] == "goal"]
        # (a) each non-start kick-off by T: exactly one goal for opp(T) since the previous kick-off.
        for idx, ko in enumerate(kickoffs):
            if idx == 0:
                continue  # the period-start kick-off never counts
            lo = kickoffs[idx - 1]["time"]
            opp = _opponent_of(ko["team"], teams)
            n = sum(1 for g in goals if lo < g["time"] <= ko["time"] and g["team"] == opp)
            assert n == 1, f"period {period} kick-off at {ko['time']} by {ko['team']}: {n} opp goals since prev"
        # (b) each goal is followed by an opponent kick-off (or period end) with no on-ball event between.
        for g in goals:
            after = [s for s in rows if s["time"] > g["time"]]
            next_ko = next((s for s in after if s["kind"] == "kickoff"), None)
            next_on_ball = next((s for s in after if s["kind"] == "on_ball"), None)
            if next_ko is None:
                assert next_on_ball is None, f"period {period} goal at {g['time']}: play continued, no kick-off"
                continue
            if next_on_ball is not None:
                assert next_on_ball["time"] >= next_ko["time"], (
                    f"period {period} goal at {g['time']}: on-ball before the kick-off"
                )
            assert next_ko["team"] == _opponent_of(g["team"], teams), (
                f"period {period} goal at {g['time']}: next kick-off not by the conceding team"
            )


def _idsse_signals(events: pd.DataFrame, actions: pd.DataFrame) -> tuple[list[dict], list[object]]:
    # Kick-offs come from the native events (SPADL drops them); goals + on-ball come from the SPADL output,
    # so a scored penalty (resolved by §25 reconciliation) counts as a goal on the same period-relative
    # clock. The two real match teams are exactly the kick-off teams ("unknown" etc. never kick off).
    teams = [t for t in events[events["event_type"] == "KickOff"]["team"].dropna().unique()]
    foul = spadlconfig.actiontype_id["foul"]
    non_action = spadlconfig.actiontype_id["non_action"]
    signals: list[dict] = []
    for _, e in events[events["event_type"] == "KickOff"].iterrows():
        signals.append(
            {"period": int(e["period"]), "time": float(e["timestamp_seconds"]), "kind": "kickoff", "team": e["team"]}
        )
    for _, a in actions.iterrows():
        period, t, team, tid = int(a["period_id"]), float(a["time_seconds"]), a["team_id"], a["type_id"]
        if tid in _GOAL_TYPES and a["result_id"] == _SUCCESS:
            signals.append({"period": period, "time": t, "kind": "goal", "team": team})
        elif a["result_id"] == _OWNGOAL:
            signals.append({"period": period, "time": t, "kind": "goal", "team": _opponent_of(team, teams)})
        elif tid not in (foul, non_action):
            signals.append({"period": period, "time": t, "kind": "on_ball", "team": team})
    return signals, teams


def test_guard1b_idsse_kickoff_reconciliation():
    # Guard 1b runs on IDSSE only (owner ruling 2026-10-21: Metrica has essentially no value and must not
    # cost CI time; the Sample Game 2 bronze fixture and Metrica Guard 1b are dropped).
    for match in IDSSE_MATCHES:
        events = pd.read_parquet(_IDSSE / f"{match}_events_native.parquet")
        actions, _ = _convert_idsse(match)
        signals, teams = _idsse_signals(events, actions)
        _check_kickoff_reconciliation(signals, teams)


def _spadl_goal_events(actions: pd.DataFrame, teams: list[object]) -> list[tuple[int, float, object]]:
    """Each SPADL goal as (period, time, credited_team): shots to the scorer, own goals to the opponent."""
    out: list[tuple[int, float, object]] = []
    goal_rows = actions[(actions["type_id"].isin(_GOAL_TYPES)) & (actions["result_id"] == _SUCCESS)]
    for _, r in goal_rows.iterrows():
        out.append((int(r["period_id"]), float(r["time_seconds"]), r["team_id"]))
    owngoals = actions[actions["result_id"] == _OWNGOAL]
    for _, r in owngoals.iterrows():
        credited = _opponent_of(r["team_id"], teams)
        if credited is not None:
            out.append((int(r["period_id"]), float(r["time_seconds"]), credited))
    return out


def test_guard1c_idsse_running_score_reconciliation():
    # Decision 3 / Guard 1c (owner-approved): at every native goal event carrying a CurrentResult, the
    # cumulative SPADL goals per team (match order) must equal it. This is the per-team, per-goal check the
    # totals gate cannot make; on DFL it covers the C1 own-goal / penalty-attribution gap.
    for match in IDSSE_MATCHES:
        events = pd.read_parquet(_IDSSE / f"{match}_events_native.parquet")
        actions, _ = _convert_idsse(match)
        teams = [t for t in events[events["event_type"] == "KickOff"]["team"].dropna().unique()]
        goals = _spadl_goal_events(actions, teams)
        native = events[
            (events["event_type"] == "ShotAtGoal")
            & (events["shot_outcome_type"].isin(["goal", "successful"]))
            & (events["shot_outcome_current_result"].notna())
        ]
        for _, ng in native.iterrows():
            cutoff = (int(ng["period"]), float(ng["timestamp_seconds"]))
            cum = {"home": 0, "away": 0}
            for period, t, team in goals:
                if (period, t) <= cutoff and str(team) in cum:
                    cum[str(team)] += 1
            home_goals, away_goals = (int(x) for x in str(ng["shot_outcome_current_result"]).split(":"))
            assert (cum["home"], cum["away"]) == (home_goals, away_goals), (match, cutoff)


# ===========================================================================
# SkillCorner opendata — Part-1 G3 reconciliation (Guard 1 coverage)
# ===========================================================================


def test_guard1_skillcorner_opendata_g3_present_and_passing():
    from silly_kicks.spadl import skillcorner

    opendata = _DATASETS / "skillcorner" / "opendata"
    match_ids = sorted(p.name.replace("_events_slim.csv", "") for p in opendata.glob("*_events_slim.csv"))
    assert match_ids
    for mid in match_ids:
        meta = json.loads((opendata / f"{mid}_match.json").read_text(encoding="utf-8"))
        events = pd.read_csv(opendata / f"{mid}_events_slim.csv", low_memory=False)
        _, report = skillcorner.convert_to_actions(events, meta)
        assert hasattr(report, "goals_coverage_gap")
        assert not report.goals_coverage_gap, mid


def test_repair_kind_emitted_subset_of_vocabulary():
    # Every repair_kind value emitted by a converter on a committed fixture is in the closed vocabulary
    # (decision D1). SkillCorner opendata exercises the Part-1 G3 classes.
    from silly_kicks.spadl import skillcorner

    emitted: set[str] = set()
    opendata = _DATASETS / "skillcorner" / "opendata"
    for mid in sorted(p.name.replace("_events_slim.csv", "") for p in opendata.glob("*_events_slim.csv")):
        meta = json.loads((opendata / f"{mid}_match.json").read_text(encoding="utf-8"))
        events = pd.read_csv(opendata / f"{mid}_events_slim.csv", low_memory=False)
        actions, _ = skillcorner.convert_to_actions(events, meta)
        emitted |= set(actions["repair_kind"].dropna().unique())
    assert emitted <= REPAIR_KIND_VALUES, emitted - REPAIR_KIND_VALUES
