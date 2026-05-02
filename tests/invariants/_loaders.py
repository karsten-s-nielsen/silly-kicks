"""Provider-agnostic SPADL loaders for the invariant test suite.

Each loader returns ``(actions, home_team_id)`` for one match using the same
converter that production code uses. Real fixtures where they exist (StatsBomb,
Sportec bronze, Metrica bronze, kloppy-Sportec/Metrica synthetic XML/JSON, PFF
synthetic). Wyscout + Opta have no real fixtures committed -- both use
synthetic 2-team event sets that exercise per-team direction.

The Wyscout 2-team fixture is in possession-perspective (each team's shot at
x=90 in raw frame) -- matches the lakehouse-side empirical observation
(99.97% of Wyscout shots at high-x in raw bronze).

The Opta synthetic is in absolute-frame, home-right, no per-period switch --
the convention silly-kicks's Opta converter expects (verified empirically by
``scripts/probe_opta_convention.py``). See PR-S22 / ADR-006.

The ``home_team_id`` is returned alongside the actions so VAEP-side tests
(``test_vaep_geometric_sanity.py``) build their game series with the SAME
home_team_id the converter used. Using a different value would silently
break VAEP's ``play_left_to_right`` away_idx computation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_KLOPPY_FIXTURES_DIR = _REPO_ROOT / "tests" / "datasets" / "kloppy"


# Returned by every loader: (SPADL actions, home_team_id used in conversion).
LoaderResult = tuple[pd.DataFrame, "int | str"]


# ---------------------------------------------------------------------------
# StatsBomb (3 real fixtures x 2 fidelity versions)
# ---------------------------------------------------------------------------


def _adapt_statsbomb_raw(raw: list[dict[str, object]], game_id: int) -> pd.DataFrame:
    _top_level_keys = {"id", "period", "timestamp", "team", "player", "type", "location"}
    return pd.DataFrame(
        [
            {
                "game_id": game_id,
                "event_id": e.get("id"),
                "period_id": e.get("period"),
                "timestamp": e.get("timestamp"),
                "team_id": (e.get("team") or {}).get("id"),
                "player_id": (e.get("player") or {}).get("id"),
                "type_name": (e.get("type") or {}).get("name"),
                "location": e.get("location"),
                "extra": {k: v for k, v in e.items() if k not in _top_level_keys},
            }
            for e in raw
        ]
    )


def load_statsbomb(match_id: int, *, xy_fidelity_version: int = 1) -> LoaderResult:
    from silly_kicks.spadl import statsbomb

    fixture_path = _REPO_ROOT / "tests" / "datasets" / "statsbomb" / "raw" / "events" / f"{match_id}.json"
    with open(fixture_path, encoding="utf-8") as f:
        raw = json.load(f)
    adapted = _adapt_statsbomb_raw(raw, game_id=match_id)
    home_team_id = int(adapted["team_id"].dropna().iloc[0])
    actions, _ = statsbomb.convert_to_actions(
        adapted,
        home_team_id=home_team_id,
        xy_fidelity_version=xy_fidelity_version,
        shot_fidelity_version=xy_fidelity_version,
    )
    return actions, home_team_id


# ---------------------------------------------------------------------------
# Wyscout — synthetic 2-team possession-perspective fixture
# ---------------------------------------------------------------------------


_WS_TYPE_SHOT = 10
_WS_SUBTYPE_SHOT_ON_TARGET = 100
_WS_TYPE_PASS = 8
_WS_SUBTYPE_SIMPLE_PASS = 85


def load_wyscout_2team_synthetic() -> LoaderResult:
    """Two-team Wyscout-shape fixture exercising shots from each team.

    Possession-perspective convention: each team's shot is at x=90 (their own
    attacking goal in raw 0-100 frame). After conversion both teams' shots
    should be at high-x (>52.5) in SPADL output.
    """
    from silly_kicks.spadl import wyscout

    HOME, AWAY = 100, 200
    rows: list[dict[str, object]] = []
    eid = 0
    for team_id in (HOME, AWAY):
        for period in (1, 2):
            for shot_num in range(6):
                eid += 1
                # Shot from raw x=85-95 toward goal x=100 (possession-perspective)
                rows.append(
                    {
                        "game_id": 1,
                        "event_id": eid,
                        "period_id": period,
                        "milliseconds": 1000.0 + eid * 100,
                        "team_id": team_id,
                        "player_id": team_id * 10 + shot_num,
                        "type_id": _WS_TYPE_SHOT,
                        "subtype_id": _WS_SUBTYPE_SHOT_ON_TARGET,
                        "positions": [{"y": 50, "x": 85 + shot_num}, {"y": 50, "x": 100}],
                        "tags": [{"id": 1801}],
                    }
                )
            # Add a pass per (team, period) for context
            eid += 1
            rows.append(
                {
                    "game_id": 1,
                    "event_id": eid,
                    "period_id": period,
                    "milliseconds": 500.0 + eid * 100,
                    "team_id": team_id,
                    "player_id": team_id * 10 + 99,
                    "type_id": _WS_TYPE_PASS,
                    "subtype_id": _WS_SUBTYPE_SIMPLE_PASS,
                    "positions": [{"y": 50, "x": 50}, {"y": 50, "x": 60}],
                    "tags": [{"id": 1801}],
                }
            )
    events = pd.DataFrame(rows)
    actions, _ = wyscout.convert_to_actions(events, home_team_id=HOME)
    return actions, HOME


# ---------------------------------------------------------------------------
# Opta — synthetic 2-team absolute_no_switch fixture (matches converter contract)
# ---------------------------------------------------------------------------


def load_opta_2team_synthetic() -> LoaderResult:
    """Two-team Opta-shape fixture in absolute-frame-no-switch convention.

    The convention silly-kicks's Opta converter expects per docstring (and per
    ``scripts/probe_opta_convention.py``). Home team always attacks right
    (x=85-95), away team always attacks left (x=5-15), in BOTH periods.

    Opta event type "miss" is the dispatch surface for a shot.
    """
    from silly_kicks.spadl import opta

    HOME, AWAY = 100, 200
    SHOT = "miss"
    rows: list[dict[str, object]] = []
    eid = 0
    for period in (1, 2):
        for shot_num in range(6):
            eid += 1
            rows.append(
                {
                    "game_id": 1,
                    "event_id": f"opta-{eid}",
                    "period_id": period,
                    "minute": shot_num + period * 10,
                    "second": 0,
                    "team_id": HOME,
                    "player_id": HOME * 10 + shot_num,
                    "type_name": SHOT,
                    "outcome": 0,
                    "start_x": 85.0 + shot_num,
                    "start_y": 50.0,
                    "end_x": 100.0,
                    "end_y": 50.0,
                    "qualifiers": {},
                }
            )
            eid += 1
            rows.append(
                {
                    "game_id": 1,
                    "event_id": f"opta-{eid}",
                    "period_id": period,
                    "minute": shot_num + period * 10 + 1,
                    "second": 0,
                    "team_id": AWAY,
                    "player_id": AWAY * 10 + shot_num,
                    "type_name": SHOT,
                    "outcome": 0,
                    "start_x": 5.0 + shot_num,
                    "start_y": 50.0,
                    "end_x": 0.0,
                    "end_y": 50.0,
                    "qualifiers": {},
                }
            )
    events = pd.DataFrame(rows)
    actions, _ = opta.convert_to_actions(events, home_team_id=HOME)
    return actions, HOME


# ---------------------------------------------------------------------------
# Sportec native (bronze parquet)
# ---------------------------------------------------------------------------


def load_sportec_native() -> LoaderResult:
    from silly_kicks.spadl import sportec

    parquet_path = _REPO_ROOT / "tests" / "datasets" / "idsse" / "sample_match.parquet"
    events = pd.read_parquet(parquet_path)
    # IDSSE bronze fixture uses literal "home"/"away" team labels in the team
    # column (not team_ids). Pass "home" so the converter mirrors only away-team
    # rows. NOTE: the existing _load_idsse_fixture in test_cross_provider_parity.py
    # uses events["team"].iloc[0] which picks "away" because the first row
    # happens to be by the away team -- fine for keeper-action tests but wrong
    # for direction-of-play tests where home_team_id correctness is load-bearing.
    home_team_id = "home"
    gk_ids: set[str] | None = None
    if "play_goal_keeper_action" in events.columns:
        gk_ids = set(events.loc[events["play_goal_keeper_action"].notna(), "player_id"].dropna().astype(str).tolist())
    actions, _ = sportec.convert_to_actions(
        events,
        home_team_id=home_team_id,
        home_team_start_left=False,  # PR-S23: same match (idsse_J03WMX) as per_period fixture; home attacks LEFT in P1
        goalkeeper_ids=gk_ids,
    )
    return actions, home_team_id


# ---------------------------------------------------------------------------
# Sportec via kloppy gateway (synthetic XML)
# ---------------------------------------------------------------------------


def load_sportec_via_kloppy() -> LoaderResult:
    from kloppy import sportec

    from silly_kicks.spadl import kloppy as kloppy_mod

    dataset = sportec.load_event(
        event_data=str(_KLOPPY_FIXTURES_DIR / "sportec_events.xml"),
        meta_data=str(_KLOPPY_FIXTURES_DIR / "sportec_meta.xml"),
    )
    home_team_id = dataset.metadata.teams[0].team_id  # Orientation.HOME_AWAY puts home first
    actions, _ = kloppy_mod.convert_to_actions(dataset, game_id="sportec_via_kloppy")
    return actions, home_team_id


# ---------------------------------------------------------------------------
# Metrica native (bronze parquet)
# ---------------------------------------------------------------------------


def load_metrica_native() -> LoaderResult:
    from silly_kicks.spadl import metrica

    parquet_path = _REPO_ROOT / "tests" / "datasets" / "metrica" / "sample_match.parquet"
    events = pd.read_parquet(parquet_path)
    home_team = str(events["team"].dropna().iloc[0])
    home_passes = events[(events["type"] == "PASS") & (events["team"] == home_team) & events["player"].notna()]
    if home_passes.empty:
        gk_ids: set[str] | None = None
    else:
        gk_ids = {str(home_passes["player"].iloc[0])}
    actions, _ = metrica.convert_to_actions(
        events,
        home_team_id=home_team,
        # PR-S23: legacy P1-only fixture; True is bit-identical to old absolute-frame behaviour
        home_team_start_left=True,
        goalkeeper_ids=gk_ids,
    )
    return actions, home_team


# ---------------------------------------------------------------------------
# Sportec native -- PER-PERIOD fixture (PR-S23 / silly-kicks 3.0.1)
# ---------------------------------------------------------------------------


def load_sportec_native_per_period() -> LoaderResult:
    """Load the dense per-period IDSSE fixture for the per-(team, period) invariant.

    Per-period orientation signature (verified empirically):
    home attacks LEFT in P1 -> home_team_start_left=False.
    """
    from silly_kicks.spadl import sportec

    parquet_path = _REPO_ROOT / "tests" / "datasets" / "idsse" / "per_period_match.parquet"
    events = pd.read_parquet(parquet_path)
    home_team_id = "home"
    gk_ids: set[str] | None = None
    if "play_goal_keeper_action" in events.columns:
        gk_ids = set(events.loc[events["play_goal_keeper_action"].notna(), "player_id"].dropna().astype(str).tolist())
    actions, _ = sportec.convert_to_actions(
        events,
        home_team_id=home_team_id,
        home_team_start_left=False,  # PR-S23: home attacks LEFT in P1 (lakehouse probe)
        goalkeeper_ids=gk_ids,
    )
    return actions, home_team_id


# ---------------------------------------------------------------------------
# Metrica native -- PER-PERIOD fixture (PR-S23 / silly-kicks 3.0.1)
# ---------------------------------------------------------------------------


def load_metrica_native_per_period() -> LoaderResult:
    """Load the dense per-period Metrica Sample Game 1 fixture.

    Per-period orientation signature (verified empirically):
    home attacks RIGHT in P1 -> home_team_start_left=True.
    """
    from silly_kicks.spadl import metrica

    parquet_path = _REPO_ROOT / "tests" / "datasets" / "metrica" / "per_period_match.parquet"
    events = pd.read_parquet(parquet_path)
    # PR-S23: hard-code "Home" rather than events["team"].iloc[0] heuristic --
    # see feedback_home_team_id_heuristic_fragile memory; first row is "Away"
    # in this fixture so the heuristic would silently invert home/away.
    home_team = "Home"
    home_passes = events[(events["type"] == "PASS") & (events["team"] == home_team) & events["player"].notna()]
    if home_passes.empty:
        gk_ids: set[str] | None = None
    else:
        gk_ids = {str(home_passes["player"].iloc[0])}
    actions, _ = metrica.convert_to_actions(
        events,
        home_team_id=home_team,
        home_team_start_left=True,  # PR-S23: home attacks RIGHT in P1 (lakehouse probe)
        goalkeeper_ids=gk_ids,
    )
    return actions, home_team


# ---------------------------------------------------------------------------
# PFF (synthetic generator output)
# ---------------------------------------------------------------------------


def load_pff_synthetic() -> LoaderResult:
    from silly_kicks.spadl import pff
    from tests.spadl.test_pff import _load_synthetic_events

    home_team_id = 100
    events = _load_synthetic_events()
    actions, _ = pff.convert_to_actions(
        events,
        home_team_id=home_team_id,
        home_team_start_left=True,
        home_team_start_left_extratime=True,
    )
    return actions, home_team_id
