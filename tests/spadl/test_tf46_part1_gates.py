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

from scripts.measure_tf46_census import expected_restart_census, expected_sportec_f4_f7_census
from silly_kicks.id_compat import canonical_id as _ci
from silly_kicks.spadl import config as spadlcfg
from silly_kicks.spadl import skillcorner, sportec, statsbomb
from silly_kicks.spadl._training_exclusion import training_exclusion_mask


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


# ---------------------------------------------------------------------------
# Sportec F4 (BallClaiming) + F7 (free-kick crossed) gates on the IDSSE fixtures (Task 3.4)
# ---------------------------------------------------------------------------
_IDSSE = Path(__file__).parents[1] / "datasets" / "idsse"

# Pinned from scripts/measure_tf46_census.py (expected_sportec_f4_f7_census, independent oracle) over the 7
# committed IDSSE fixtures, 2026-10-20. Fixture SHAs: tests/datasets/tf46_fixture_sha256sums.txt.
_PIN_BALL_CLAIM_COUNTS = {"BallClaimed": 42, "BallHeld": 17, "Block": 1, "InterceptedBall": 146}
# interceptions_added = 134 InterceptedBall + 42 BallClaimed. Spec section 7.1 amended 2026-10-20:
# BallClaimed is an outfield ball win, mapped to a derived interception (house precedent: Metrica RECOVERY,
# SkillCorner "recovery"). Every claim now lands -> ball_claim_unattached == 0 (Block's one partner survives).
_PIN_INTERCEPTIONS_ADDED = 176
_PIN_KEEPER_PICK_UPS_ADDED = 4
_PIN_BALL_CLAIM_UNATTACHED = 0
_PIN_FREEKICK_CROSSED = 48
_PIN_FREEKICK_SHORT = 124
_PIN_FREEKICK_INFERRED = 4


def _idsse_fixtures() -> list[tuple[str, pd.DataFrame, dict]]:
    out = []
    for pq in sorted(_IDSSE.glob("*_events_native.parquet")):
        bare = pq.name.replace("_events_native.parquet", "")
        meta = json.loads((_IDSSE / f"{bare}_meta.json").read_text(encoding="utf-8"))
        out.append((bare, pd.read_parquet(pq), meta))
    return out


@lru_cache(maxsize=1)
def _sportec_converted() -> tuple:
    out = []
    for bare, ev, meta in _idsse_fixtures():
        # Orientation key is the role literal "home", NOT meta["home_team_id"] (the DFL
        # club id). The IDSSE bronze `team` column holds {"home","away"} and the converter
        # maps `team` -> `team_id` verbatim (sportec.py), so _mirror_per_period keys is_home
        # on `team_id == "home"`. Passing the club id leaves is_home all-False, so the home
        # team is never mirrored to LTR and the ADR-006 direction-of-play invariant
        # (SILLY_KICKS_ASSERT_INVARIANTS=1, set by CI) fails. Matches the canonical sportec
        # per-period loader (tests/invariants/_loaders.py). The per-fixture regular/ET start
        # flags still come from the meta.
        actions, report = sportec.convert_to_actions(
            ev,
            home_team_id="home",
            home_team_start_left=meta["home_team_start_left"],
            home_team_start_left_extratime=meta.get("home_team_start_left_extratime"),
        )
        out.append((bare, actions, report))
    return tuple(out)


def test_sportec_f4_f7_pins_match_independent_oracle():
    # Ties the pinned F4/F7 constants to the INDEPENDENT oracle (raw-event re-derivation, no converter).
    claims: Counter = Counter()
    interceptions = keeper = unattached = crossed = short = inferred = 0
    for _bare, ev, _meta in _idsse_fixtures():
        c = expected_sportec_f4_f7_census(ev)
        claims.update(c["ball_claim_counts"])
        interceptions += c["interceptions_added"]
        keeper += c["keeper_pick_ups_added"]
        unattached += c["ball_claim_unattached"]
        crossed += c["freekick_crossed"]
        short += c["freekick_short"]
        inferred += c["freekick_inferred"]
    assert dict(claims) == _PIN_BALL_CLAIM_COUNTS
    assert interceptions == _PIN_INTERCEPTIONS_ADDED
    assert keeper == _PIN_KEEPER_PICK_UPS_ADDED
    assert unattached == _PIN_BALL_CLAIM_UNATTACHED
    assert (crossed, short, inferred) == (_PIN_FREEKICK_CROSSED, _PIN_FREEKICK_SHORT, _PIN_FREEKICK_INFERRED)


def test_g6_sportec_f4_converter_matches_pins():
    claims: Counter = Counter()
    report_interceptions = report_keeper = report_unattached = 0
    stream_interceptions = stream_keeper = 0
    dup_keeper_rows = 0
    bad_token_rows = 0
    valid_tokens = {"InterceptedBall", "BallClaimed", "BallHeld", "Block"}
    keeper_ids = {spadlcfg.actiontype_id[k] for k in ("keeper_save", "keeper_claim", "keeper_punch", "keeper_pick_up")}
    for _bare, actions, report in _sportec_converted():
        claims.update(report.ball_claim_counts)
        report_interceptions += report.interceptions_added
        report_keeper += report.keeper_pick_ups_added
        report_unattached += report.ball_claim_unattached
        oeid = actions["original_event_id"].astype(str)
        stream_interceptions += int(oeid.str.endswith("_f4_interception").sum())
        f4_kpu = actions[oeid.str.endswith("_f4_keeper_pick_up")]
        stream_keeper += len(f4_kpu)
        bct = actions["ball_claim_type"]
        bad_token_rows += int((bct.notna() & ~bct.isin(valid_tokens)).sum())
        # Duplicate-keeper guard: no F4 keeper_pick_up coincides with another keeper action (same player, <=0.1 s).
        keepers = actions[actions["type_id"].isin(keeper_ids)]
        for _, row in f4_kpu.iterrows():
            same = keepers[
                (keepers["player_id"].astype(str) == str(row["player_id"]))
                & (keepers["period_id"] == row["period_id"])
                & ((keepers["time_seconds"] - row["time_seconds"]).abs() <= 0.1)
                & (keepers["original_event_id"].astype(str) != str(row["original_event_id"]))
            ]
            dup_keeper_rows += len(same)
    # The converter's own audit matches the pins ...
    assert dict(claims) == _PIN_BALL_CLAIM_COUNTS
    assert report_interceptions == _PIN_INTERCEPTIONS_ADDED
    assert report_keeper == _PIN_KEEPER_PICK_UPS_ADDED
    assert report_unattached == _PIN_BALL_CLAIM_UNATTACHED
    # ... and so does an independent re-count of the actual SPADL stream (not just the report).
    assert stream_interceptions == _PIN_INTERCEPTIONS_ADDED
    assert stream_keeper == _PIN_KEEPER_PICK_UPS_ADDED
    assert bad_token_rows == 0  # ball_claim_type only ever carries the 4 DFL claim_type tokens
    assert dup_keeper_rows == 0  # the duplicate-keeper guard held on every F4 keeper_pick_up


def test_g7_sportec_f7_converter_matches_pins():
    crossed = short = inferred = pen_box_true = 0
    for _bare, ev, _meta in _idsse_fixtures():
        pb = ev.loc[ev["event_type"] == "FreeKick", "play_penalty_box"].astype(str).str.lower()
        pen_box_true += int((pb == "true").sum())
    for _bare, actions, _report in _sportec_converted():
        tn = actions["type_id"].map(_ID_TO_NAME)
        crossed += int((tn == "freekick_crossed").sum())
        short += int((tn == "freekick_short").sum())
        # A missing-qualifier free kick stays freekick_short with type_source="inferred".
        inferred += int(((tn == "freekick_short") & (actions["type_source"] == "inferred")).sum())
    assert crossed == pen_box_true == _PIN_FREEKICK_CROSSED  # crossed count equals the play_penalty_box true count
    assert short == _PIN_FREEKICK_SHORT
    assert inferred == _PIN_FREEKICK_INFERRED


# ---------------------------------------------------------------------------
# StatsBomb K2 (G8) + shared exclusion helper (G-train) — Commit 4 (spec section 8/11.3)
# ---------------------------------------------------------------------------
_STATSBOMB = Path(__file__).parents[1] / "datasets" / "statsbomb"


def _k2_synthetic_converted():
    events = pd.DataFrame(json.loads((_STATSBOMB / "k2_synthetic.json").read_text(encoding="utf-8")))
    actions, _ = statsbomb.convert_to_actions(events, home_team_id=100, xy_fidelity_version=1, shot_fidelity_version=1)
    out = actions.copy()
    out["type_name"] = out["type_id"].map(_ID_TO_NAME)
    return out.set_index(out["original_event_id"].astype(str))


_K2_SETPIECE_IDS = {
    spadlcfg.actiontype_id[k]
    for k in ("corner_crossed", "corner_short", "freekick_crossed", "freekick_short", "goalkick", "throw_in")
}


def test_g8_statsbomb_k2_on_committed_fixture():
    # G8: a set-piece pass with Unknown outcome is kept + typed; a non-set-piece Unknown pass stays
    # non_action; chain rows follow the section-8.2 convention. Reads the committed synthetic fixture.
    by = _k2_synthetic_converted()
    assert by.loc["c1_corner", "type_name"] in ("corner_crossed", "corner_short")
    assert by.loc["c2_fk", "type_name"] in ("freekick_crossed", "freekick_short")
    assert by.loc["c3_gk", "type_name"] == "goalkick"
    assert by.loc["c4_ti", "type_name"] == "throw_in"
    assert by.loc["c5_corner", "type_name"] in ("corner_crossed", "corner_short")
    for evid in ("c1_corner", "c2_fk", "c3_gk", "c4_ti", "c5_corner"):
        assert by.loc[evid, "result_source"] == "inferred", evid
    assert "e_nonsp" not in by.index  # non-set-piece Unknown stays non_action (dropped)
    # chain: corner->FK (c1), FK->penalty (c2), goal kick -> Foul Committed (skipped) -> FK through a foul (c3)
    for evid in ("c1_corner", "c2_fk", "c3_gk"):
        assert by.loc[evid, "result_id"] == spadlcfg.result_id["success"], evid
        assert by.loc[evid, "restart_evidence"] == "chain_ambiguous", evid


def test_g8_statsbomb_real_open_data_e2e():
    # Real-open-data e2e (decision A5(1)): match 7584 keeps >= 1 set-piece pass with result_source="inferred"
    # (the K2 fix; this match has a kept goal kick). load_statsbomb reads the COMMITTED 7584.json (no network,
    # no statsbombpy), so this is a hard gate -- a converter crash fails it, it is NOT skipped.
    from tests.invariants._loaders import load_statsbomb

    actions, _ = load_statsbomb(7584)
    kept = actions[actions["result_source"] == "inferred"]
    # Pinned on the committed 7584.json (TF46C-IMPL-22/-21/-27): exactly one kept set-piece-Unknown pass -- a
    # goal kick whose next MEANINGFUL event WITHIN THE HALF (bookkeeping / Foul Committed-Won skipped) is a
    # same-team free kick, so the §8.2 chain convention fires on real data: success + chain_ambiguous. This
    # pins the chain actually firing (the whole point of the -21 successor-scan fix), not just >= 1 kept.
    assert len(kept) == 1
    kept_row = kept.iloc[0]
    assert kept_row["type_id"] == spadlcfg.actiontype_id["goalkick"]
    assert kept_row["result_id"] == spadlcfg.result_id["success"]
    assert kept_row["restart_evidence"] == "chain_ambiguous"
    assert kept["type_id"].isin(_K2_SETPIECE_IDS).all()
    assert int((actions["restart_evidence"] == "chain_ambiguous").sum()) == 1


def test_g_train_helper_excludes_the_four_classes():
    # G-train: the shared helper excludes inferred/stopgap result_source, chain_ambiguous restart_evidence
    # and any repair_kind row; it keeps native and <NA>. (The xsuccess + expected_passing filters call it --
    # tests/xsuccess + tests/expected_passing assert the exclusion is effective in a fit.)
    df = pd.DataFrame(
        {
            "result_source": ["inferred", "stopgap", pd.NA, pd.NA, "native"],
            "restart_evidence": [pd.NA, pd.NA, "chain_ambiguous", pd.NA, pd.NA],
            "repair_kind": [pd.NA, pd.NA, pd.NA, "own_goal_inverted", pd.NA],
        }
    )
    assert training_exclusion_mask(df).tolist() == [True, True, True, True, False]
