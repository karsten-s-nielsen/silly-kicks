"""SkillCorner keeper-appearance extractor (TF-59 PR1, spec §5.5).

The extractor reads a parsed SkillCorner ``match.json`` dict -- ``players[].playing_time.by_period[]``
(``start_frame`` / ``end_frame``) + ``match_periods`` -- identifies keepers by
``player_role.acronym == "GK"`` (equivalently ``player_role.name == "Goalkeeper"``; see the
INVESTIGATION note below), converts frames -> period-relative seconds via the per-period frame offsets,
builds ONE :class:`~silly_kicks.keeper_identity.KeeperSegment` per keeper (first -> last on-pitch
period span, ``source="native_intervals"``) and delegates to the shared Task-6 builder.

The committed happy-path fixture ``tests/datasets/skillcorner/public_match.json`` is the real PUBLIC
A-League match ``1886347`` (redistributable open data, fetched via the public pining token). It carries
two full-match starting keepers and NO GK change (probe-verified: 0 GK subs among the 10 public
SkillCorner matches), so the unit suite exercises the happy path here and a SYNTHETIC by_period GK
change exercises the mid-match keeper-change interval math.

INVESTIGATION (real peggy44 GK-change matches, HF ``peggy44/RealMadrid24-25``, 2026-09-01):
``player_role.position_group`` is **always** ``"Other"`` for a keeper -- it does NOT identify keepers.
But ``acronym == "GK"`` / ``name == "Goalkeeper"`` DOES catch a SUBSTITUTE keeper: in all three
GK-change matches inspected the incoming keeper is tagged ``{acronym: "GK", name: "Goalkeeper",
position_group: "Other"}`` (n_gk_acronym == 3 == 2 starters + 1 sub keeper). So there is NO sub-keeper
*identification* gap; the identification rule ``acronym == "GK"`` catches starters AND sub keepers.
The separate INTERVAL gap is a schema divergence (see the e2e): the peggy44 export carries no
``match_periods`` / ``by_period`` (only clock ``start_time`` / ``end_time``), so the by_period
extractor yields no intervals there -- interval extraction requires the by_period-bearing A-League
schema.
"""

from __future__ import annotations

import json
import math
import pathlib
import warnings

import numpy as np
import pytest

from silly_kicks.keeper_identity import validate_keeper_appearances
from silly_kicks.providers.skillcorner.appearances import extract_keeper_appearances

FIX = pathlib.Path(__file__).parents[2] / "datasets/skillcorner/public_match.json"


def _match() -> dict:
    with open(FIX, encoding="utf-8") as fh:
        return json.load(fh)


# --- happy path: the committed public A-League fixture (1886347) -----------------------------------


def test_public_match_two_keepers_per_period() -> None:
    ap = extract_keeper_appearances(_match())
    # two starting keepers, each with a row per period they played (no GK sub in the public 10)
    assert ap[ap["start_time_seconds"] == 0.0]["team_id"].nunique() == 2
    assert set(ap["period_id"]) <= {1, 2, 3, 4, 5}
    assert not ap["end_time_seconds"].isna().any()  # every row has a real end (finite or inf)


def test_public_match_two_full_keepers() -> None:
    ap = extract_keeper_appearances(_match())
    starters = ap[ap["start_time_seconds"] == 0.0]
    assert starters["team_id"].nunique() == 2
    # every row ends strictly after it starts (a full-match keeper's rows are open to the whistle: inf)
    assert (ap["end_time_seconds"] > 0).all() and not ap["end_time_seconds"].isna().any()


def test_periods_are_period_relative() -> None:
    ap = extract_keeper_appearances(_match())
    # each period's starter opens at 0.0 (start_frame maps to that period's start)
    for p in ap["period_id"].unique():
        assert (ap[(ap["period_id"] == p) & (ap["start_time_seconds"] == 0.0)]).shape[0] >= 1


def test_frames_convert_to_period_relative_seconds() -> None:
    ap = extract_keeper_appearances(_match())
    # both period-1 starters open at 0.0 (the period start_frame maps to 0.0)
    p1 = ap[(ap["period_id"] == 1) & (ap["start_time_seconds"] == 0.0)]
    assert len(p1) == 2


def test_public_match_shape_and_source() -> None:
    ap = extract_keeper_appearances(_match())
    # 2 full-match keepers x 2 periods -> 4 rows, all native_intervals, both teams present in each period.
    assert len(ap) == 4
    assert (ap["source"] == "native_intervals").all()
    assert set(ap["period_id"]) == {1, 2}
    for p in (1, 2):
        assert ap[ap["period_id"] == p]["team_id"].nunique() == 2
    # a full-match keeper's rows are open to the whistle (inf), never NaN.
    assert bool(np.isinf(ap["end_time_seconds"]).all())
    validate_keeper_appearances(ap)


def test_game_id_defaults_to_match_id() -> None:
    m = _match()
    ap = extract_keeper_appearances(m)
    assert (ap["game_id"] == m["id"]).all()
    # explicit game_id overrides the default
    ap2 = extract_keeper_appearances(m, game_id="override")
    assert (ap2["game_id"] == "override").all()


def test_pure_input_not_mutated() -> None:
    m = _match()
    snapshot = json.loads(json.dumps(m))
    extract_keeper_appearances(m)
    assert m == snapshot


# --- synthetic by_period GK change (mid-match keeper substitution interval math) -------------------


def _synthetic_gk_change_match() -> dict:
    """A 2-period by_period-schema match: team T1 keeper A subbed for B at period-2 t=600s; T2 keeper C
    plays throughout. fps = 27000 / (45.0 * 60) = 10.0. Bench SUB (empty by_period) must be excluded."""
    return {
        "id": "syn1",
        "match_periods": [
            {
                "period": 1,
                "name": "period_1",
                "start_frame": 0,
                "end_frame": 27000,
                "duration_frames": 27000,
                "duration_minutes": 45.0,
            },
            {
                "period": 2,
                "name": "period_2",
                "start_frame": 27000,
                "end_frame": 54000,
                "duration_frames": 27000,
                "duration_minutes": 45.0,
            },
        ],
        "players": [
            {  # A: starter T1, subbed off at period-2 frame 33000 (t=600s)
                "id": "A",
                "team_id": "T1",
                "player_role": {"acronym": "GK", "name": "Goalkeeper", "position_group": "Other"},
                "playing_time": {
                    "by_period": [
                        {"name": "period_1", "start_frame": 0, "end_frame": 27000},
                        {"name": "period_2", "start_frame": 27000, "end_frame": 33000},
                    ]
                },
            },
            {  # B: sub keeper T1, on at period-2 frame 33000 (t=600s), plays to the whistle
                "id": "B",
                "team_id": "T1",
                "player_role": {"acronym": "GK", "name": "Goalkeeper", "position_group": "Other"},
                "playing_time": {
                    "by_period": [
                        {"name": "period_2", "start_frame": 33000, "end_frame": 53999},
                    ]
                },
            },
            {  # C: T2 keeper, full match
                "id": "C",
                "team_id": "T2",
                "player_role": {"acronym": "GK", "name": "Goalkeeper", "position_group": "Other"},
                "playing_time": {
                    "by_period": [
                        {"name": "period_1", "start_frame": 0, "end_frame": 27000},
                        {"name": "period_2", "start_frame": 27000, "end_frame": 53999},
                    ]
                },
            },
            {  # an unused bench substitute -- acronym SUB, empty by_period -> excluded
                "id": "Z",
                "team_id": "T2",
                "player_role": {"acronym": "SUB", "name": "Substitute", "position_group": "Other"},
                "playing_time": {"by_period": []},
            },
        ],
    }


def test_synthetic_gk_change_splits_starter_and_replacement() -> None:
    ap = extract_keeper_appearances(_synthetic_gk_change_match())
    t1 = {str(x) for x in ap[ap["team_id"] == "T1"]["player_id"]}
    t2 = {str(x) for x in ap[ap["team_id"] == "T2"]["player_id"]}
    assert t1 == {"A", "B"}  # starter then replacement
    assert t2 == {"C"}  # bench SUB excluded (empty by_period, not a keeper acronym)
    assert (ap["source"] == "native_intervals").all()

    # A's period-2 tenure ends exactly at the sub time (600s); B opens there, open to the whistle.
    a_p2 = ap[(ap["player_id"] == "A") & (ap["period_id"] == 2)].iloc[0]
    b_p2 = ap[(ap["player_id"] == "B") & (ap["period_id"] == 2)].iloc[0]
    assert a_p2["end_time_seconds"] == 600.0
    assert b_p2["start_time_seconds"] == 600.0
    assert math.isinf(b_p2["end_time_seconds"])
    # B has no period-1 row (came on in period 2); A opens period 1 at 0.0.
    assert ap[(ap["player_id"] == "B") & (ap["period_id"] == 1)].empty
    assert ap[(ap["player_id"] == "A") & (ap["period_id"] == 1)].iloc[0]["start_time_seconds"] == 0.0
    validate_keeper_appearances(ap)


def test_reduced_schema_without_by_period_is_empty_and_valid() -> None:
    # A reduced-schema match.json (no match_periods / by_period -- the peggy44 export shape) yields NO
    # by_period intervals; the extractor returns a valid EMPTY appearances frame (documented gap).
    reduced = {
        "id": "reduced1",
        "players": [
            {
                "id": "A",
                "team_id": "T1",
                "player_role": {"acronym": "GK", "name": "Goalkeeper", "position_group": "Other"},
                "start_time": "00:00:00",
                "end_time": None,
                "playing_time": {},
            },
        ],
    }
    ap = extract_keeper_appearances(reduced)
    assert ap.empty
    validate_keeper_appearances(ap)


# --- per-period fps (D2): each period is converted with its OWN derived rate --------------------


def _synthetic_two_fps_match() -> dict:
    """A 2-period match sampled at genuinely different (within-band) rates: period 1 @ 10.0 fps
    (27000 / 45.0 min) and period 2 @ 10.4 fps (28080 / 45.0 min). Keeper A (T1) is subbed off in
    period 2 at frame 28040 -> a FINITE period-2 end that must use period-2's own fps."""
    return {
        "id": "synfps",
        "match_periods": [
            {
                "period": 1,
                "name": "period_1",
                "start_frame": 0,
                "end_frame": 27000,
                "duration_frames": 27000,
                "duration_minutes": 45.0,
            },  # fps 10.0
            {
                "period": 2,
                "name": "period_2",
                "start_frame": 27000,
                "end_frame": 54000,
                "duration_frames": 28080,
                "duration_minutes": 45.0,
            },  # fps 10.4
        ],
        "players": [
            {  # A: T1 keeper, off in period 2 at frame 28040 -> finite end via period-2 fps
                "id": "A",
                "team_id": "T1",
                "player_role": {"acronym": "GK", "name": "Goalkeeper", "position_group": "Other"},
                "playing_time": {
                    "by_period": [
                        {"name": "period_1", "start_frame": 0, "end_frame": 27000},
                        {"name": "period_2", "start_frame": 27000, "end_frame": 28040},
                    ]
                },
            },
            {  # B: T1 sub keeper, on at frame 28040, plays to the whistle
                "id": "B",
                "team_id": "T1",
                "player_role": {"acronym": "GK", "name": "Goalkeeper", "position_group": "Other"},
                "playing_time": {
                    "by_period": [
                        {"name": "period_2", "start_frame": 28040, "end_frame": 53999},
                    ]
                },
            },
            {  # C: T2 keeper, full match
                "id": "C",
                "team_id": "T2",
                "player_role": {"acronym": "GK", "name": "Goalkeeper", "position_group": "Other"},
                "playing_time": {
                    "by_period": [
                        {"name": "period_1", "start_frame": 0, "end_frame": 27000},
                        {"name": "period_2", "start_frame": 27000, "end_frame": 53999},
                    ]
                },
            },
        ],
    }


def test_per_period_fps_converts_each_period_with_its_own_rate() -> None:
    # Period 2 is sampled at 10.4 fps, period 1 at 10.0. A's period-2 tenure ends at frame 28040:
    # (28040 - 27000) / 10.4 = 100.0s -- NOT the 104.0s the old single-fps (period-1 rate) path gave.
    with pytest.warns(UserWarning, match="fps differs across periods"):
        ap = extract_keeper_appearances(_synthetic_two_fps_match())
    a_p2 = ap[(ap["player_id"] == "A") & (ap["period_id"] == 2)].iloc[0]
    assert a_p2["end_time_seconds"] == pytest.approx(100.0)
    # The stale single-fps result would have been 104.0 -- assert we are NOT that (the D2 defect).
    assert a_p2["end_time_seconds"] != pytest.approx(104.0)
    # B opens exactly where A closes (same frame, same period-2 rate).
    b_p2 = ap[(ap["player_id"] == "B") & (ap["period_id"] == 2)].iloc[0]
    assert b_p2["start_time_seconds"] == pytest.approx(100.0)
    validate_keeper_appearances(ap)


def test_consistent_fps_across_periods_does_not_warn() -> None:
    # A consistent-10fps match (both periods 27000 / 45.0) must NOT emit the per-period fps warning --
    # the spread is 0, well below the warn threshold.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ap = extract_keeper_appearances(_synthetic_gk_change_match())
    fps_warnings = [w for w in caught if "fps differs across periods" in str(w.message)]
    assert not fps_warnings
    assert not ap.empty


# --- e2e: real peggy44 GK-change match via HF (confirms the identification finding + the schema gap) --


@pytest.mark.e2e
def test_peggy44_gk_change_sub_keeper_identifiable() -> None:
    hf = pytest.importorskip("huggingface_hub")
    repo, meta_id = "peggy44/RealMadrid24-25", "1287842"  # a known GK-change match (subbed-off GK id 8888)
    try:
        path = hf.hf_hub_download(repo, f"meta/{meta_id}.json", repo_type="dataset")
    except Exception as exc:  # no HF token / repo unreachable -> skip (licensed, CC-BY-NC)
        pytest.skip(f"peggy44 meta unreachable: {type(exc).__name__}")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    players = data.get("players", [])
    gk_acr = [p for p in players if (p.get("player_role") or {}).get("acronym") == "GK"]
    # A GK change means >2 acronym==GK players (starters + the sub keeper), and NONE is position_group GK.
    assert len(gk_acr) >= 3, "expected a GK-change match (>=3 acronym==GK players)"
    assert all((p.get("player_role") or {}).get("position_group") != "Goalkeeper" for p in gk_acr)
    # The incoming keeper (start_time != kickoff) is tagged acronym==GK / name==Goalkeeper, NOT "SUB".
    incoming = [p for p in gk_acr if p.get("start_time") not in (None, "", "00:00:00")]
    assert incoming, "expected an incoming (subbed-on) keeper"
    for p in incoming:
        role = p.get("player_role") or {}
        assert role.get("acronym") == "GK" and role.get("name") == "Goalkeeper"

    # The peggy44 export carries no match_periods/by_period, so the by_period extractor yields an empty
    # (but valid) appearances frame -- the documented interval gap for that reduced schema.
    ap = extract_keeper_appearances(data)
    validate_keeper_appearances(ap)
    assert ap.empty
