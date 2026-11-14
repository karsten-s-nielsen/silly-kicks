"""One delivery scene in every converter's native format (spec 2026-11-13-cross-definition §7.1, ADR-133).

The scene is written once in SPADL metres, in the ACTING team's left-to-right frame. Each builder writes it in
its provider's native schema, units and orientation convention:

* possession perspective (StatsBomb, Wyscout, SkillCorner): native coordinates are the acting team's frame;
* absolute frame, home attacks right in every period (Opta, kloppy ``HOME_AWAY``): the away team is mirrored;
* per-period absolute (Metrica, Sportec, Gradient Sports): a team is mirrored when it attacks right-to-left in
  that period, which depends on ``home_starts_left``.

So the away rows and the period-2 rows reach the delivery rule only through each converter's own orientation
handling, and a rule applied in the wrong frame (before ``to_spadl_ltr``) types them wrongly.

Scene (event id: what it is -> what the rule must make of it):

* 1: home native CROSS from wide at x=90, delivered short (next: a home pass outside the box) -> ``pass``;
* 2: that short home pass -> ``pass``;
* 3: home native PASS from wide, next a home shot in the box -> ``cross``;
* 4: the shot;
* 5: home native CROSSED corner, ended by an away goal kick (a restart) -> ``corner_short``;
* 6: the away goal kick (its start mirrors into the home box: only the restart rule stops a delivery);
* 7: away native PASS from wide, next an away shot in the box -> ``cross`` (the away team's direction);
* 8: the shot;
* 9: home native PASS from wide in PERIOD 2, next a home shot in the box -> ``cross`` (the period-2 direction);
* 10: the shot.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import json
import pathlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

_DATASETS = pathlib.Path(__file__).resolve().parents[1] / "datasets"
L, W = 105.0, 68.0

#: (event id, period, seconds, team "H"/"A", kind, x, y) -- x, y in the acting team's LTR frame, SPADL metres.
SCENE: tuple[tuple[int, int, float, str, str, float, float], ...] = (
    (1, 1, 10.0, "H", "cross", 90.0, 5.0),
    (2, 1, 12.0, "H", "pass", 92.0, 8.0),
    (3, 1, 20.0, "H", "pass", 90.0, 63.0),
    (4, 1, 22.0, "H", "shot", 95.0, 34.0),
    (5, 1, 30.0, "H", "corner_crossed", 104.0, 1.0),
    (6, 1, 40.0, "A", "goalkick", 5.5, 34.0),
    (7, 1, 50.0, "A", "pass", 90.0, 5.0),
    (8, 1, 52.0, "A", "shot", 95.0, 34.0),
    (9, 2, 10.0, "H", "pass", 90.0, 63.0),
    (10, 2, 12.0, "H", "shot", 95.0, 34.0),
)

#: The native cross a ``blocked=True`` builder marks as blocked (Wyscout tag 2101, GS cross outcome "B").
BLOCKED_EID = 1

#: The SPADL type the delivery rule must give each scene event.
EXPECTED_TYPE: dict[int, str] = {
    1: "pass",
    2: "pass",
    3: "cross",
    4: "shot",
    5: "corner_short",
    6: "goalkick",
    7: "cross",
    8: "shot",
    9: "cross",
    10: "shot",
}

#: The provider's own label (``cross_native``) on the delivery-class scene events, where the provider HAS one.
_NATIVE: dict[int, bool] = {1: True, 2: False, 3: False, 5: True, 7: False, 9: False}


def expected_cross_native(provider: str) -> dict[int, object]:
    """``cross_native`` per scene event: the native label, ``<NA>`` where the provider has no split."""
    out: dict[int, object] = {eid: pd.NA for eid, *_ in SCENE}
    if provider == "skillcorner":  # no native cross label at all
        return out
    out.update(_NATIVE)
    if provider == "metrica":  # Metrica set pieces are positional: no crossed/short split
        out[5] = pd.NA
    return out


def _attacks_right(team: str, period: int, home_starts_left: bool) -> bool:
    return (team == "H") == home_starts_left if period == 1 else (team == "H") != home_starts_left


def _absolute(x: float, y: float, team: str, period: int, convention: str, home_starts_left: bool):
    """The scene point in the provider's frame (still SPADL metres)."""
    if convention == "possession":
        return x, y
    if convention == "home_right":
        right = team == "H"
    else:  # per-period absolute
        right = _attacks_right(team, period, home_starts_left)
    return (x, y) if right else (L - x, W - y)


def _ends() -> dict[int, tuple[float, float]]:
    """Each event's end: the next event's start (same frame), so no converter inserts a carry; a shot ends at goal."""
    out: dict[int, tuple[float, float]] = {}
    for i, (eid, _period, _t, team, kind, _x, _y) in enumerate(SCENE):
        if kind == "shot":
            out[eid] = (L, W / 2)
            continue
        nxt = SCENE[i + 1]
        nx, ny = nxt[5], nxt[6]
        out[eid] = (nx, ny) if nxt[3] == team else (L - nx, W - ny)
    return out


# ---------------------------------------------------------------------------------------------------------- statsbomb
def _statsbomb(home_starts_left: bool = True) -> pd.DataFrame:
    ends = _ends()

    def sb(x: float, y: float) -> list[float]:
        return [x * 120.0 / L, (W - y) * 80.0 / W]

    rows = []
    for eid, period, t, team, kind, x, y in SCENE:
        ex, ey = ends[eid]
        base: dict[str, Any] = {
            "game_id": 1,
            "event_id": f"e{eid}",
            "period_id": period,
            "timestamp": f"00:00:{t:06.3f}",
            "team_id": 100 if team == "H" else 200,
            "player_id": 1000 + eid,
            "location": sb(x, y),
        }
        if kind == "shot":
            base |= {
                "type_name": "Shot",
                "extra": {
                    "shot": {
                        "end_location": [120.0, 40.0],
                        "outcome": {"name": "Saved"},
                        "type": {"name": "Open Play"},
                        "body_part": {"name": "Right Foot"},
                    }
                },
            }
        else:
            p: dict[str, Any] = {"end_location": sb(ex, ey), "height": {"name": "Ground Pass"}}
            if kind == "cross":
                p |= {"cross": True, "height": {"name": "High Pass"}}
            elif kind == "corner_crossed":
                p |= {"type": {"name": "Corner"}, "height": {"name": "High Pass"}}
            elif kind == "goalkick":
                p |= {"type": {"name": "Goal Kick"}}
            base |= {"type_name": "Pass", "extra": {"pass": p}}
        rows.append(base)
    return pd.DataFrame(rows)


def _statsbomb_run(events: pd.DataFrame, home_starts_left: bool = True) -> pd.DataFrame:
    from silly_kicks.spadl import statsbomb

    return statsbomb.convert_to_actions(events, home_team_id=100)[0]


# --------------------------------------------------------------------------------------------------------------- opta
def _opta(home_starts_left: bool = True) -> pd.DataFrame:
    ends = _ends()
    quals = {"pass": {}, "cross": {2: True}, "corner_crossed": {6: True, 2: True}, "goalkick": {124: True}}
    rows = []
    for eid, period, t, team, kind, x, y in SCENE:
        ax, ay = _absolute(x, y, team, period, "home_right", home_starts_left)
        ex, ey = _absolute(*ends[eid], team, period, "home_right", home_starts_left)
        rows.append(
            {
                "game_id": 1,
                "event_id": eid,
                "type_id": 13 if kind == "shot" else 1,
                "period_id": period,
                "minute": 0 if period == 1 else 45,
                "second": int(t),
                "team_id": 10 if team == "H" else 20,
                "player_id": 1000 + eid,
                "outcome": True,
                "start_x": ax / L * 100.0,
                "start_y": ay / W * 100.0,
                "end_x": ex / L * 100.0,
                "end_y": ey / W * 100.0,
                "qualifiers": {} if kind == "shot" else quals[kind],
                "type_name": "attempt saved" if kind == "shot" else "pass",
            }
        )
    return pd.DataFrame(rows)


def _opta_run(events: pd.DataFrame, home_starts_left: bool = True) -> pd.DataFrame:
    from silly_kicks.spadl import opta

    return opta.convert_to_actions(events, home_team_id=10)[0]


# ------------------------------------------------------------------------------------------------------------ wyscout
def _wyscout(home_starts_left: bool = True, *, blocked: bool = False) -> pd.DataFrame:
    ends = _ends()
    kinds = {
        "pass": (8, 85, "Simple pass", "Pass", [1801]),
        "cross": (8, 80, "Cross", "Pass", [1801]),
        "shot": (10, 100, "Shot", "Shot", [1801]),
        "corner_crossed": (3, 30, "Corner", "Free Kick", [801, 1801]),
        "goalkick": (3, 34, "Goal kick", "Free Kick", [1801]),
    }
    rows = []
    for eid, period, t, team, kind, x, y in SCENE:
        type_id, subtype_id, subtype_name, type_name, tags = kinds[kind]
        ex, ey = ends[eid]
        rows.append(
            {
                "type_id": type_id,
                "subtype_name": subtype_name,
                "subtype_id": subtype_id,
                "tags": [{"id": tag} for tag in (*tags, *((2101,) if blocked and eid == BLOCKED_EID else ()))],
                "player_id": 1000 + eid,
                "positions": [
                    {"x": x / L * 100.0, "y": 100.0 - y / W * 100.0},
                    {"x": ex / L * 100.0, "y": 100.0 - ey / W * 100.0},
                ],
                "game_id": 1,
                "type_name": type_name,
                "team_id": 100 if team == "H" else 200,
                "period_id": period,
                "milliseconds": t * 1000.0,
                "event_id": eid,
            }
        )
    return pd.DataFrame(rows)


def _wyscout_run(events: pd.DataFrame, home_starts_left: bool = True) -> pd.DataFrame:
    from silly_kicks.spadl import wyscout

    return wyscout.convert_to_actions(events, home_team_id=100)[0]


# ------------------------------------------------------------------------------------------------------------ metrica
def _metrica(home_starts_left: bool = True) -> pd.DataFrame:
    ends = _ends()
    kinds = {
        "pass": ("PASS", None),
        "cross": ("PASS", "CROSS"),
        "shot": ("SHOT", "ON TARGET-SAVED"),
        "corner_crossed": ("SET PIECE", "CORNER KICK"),
        "goalkick": ("SET PIECE", "GOAL KICK"),
    }
    rows = []
    for eid, period, t, team, kind, x, y in SCENE:
        typ, sub = kinds[kind]
        ax, ay = _absolute(x, y, team, period, "per_period", home_starts_left)
        ex, ey = _absolute(*ends[eid], team, period, "per_period", home_starts_left)
        rows.append(
            {
                "match_id": "G1",
                "event_id": f"e{eid}",
                "type": typ,
                "subtype": sub,
                "period": period,
                "start_time_s": t + (0.0 if period == 1 else 2700.0),
                "end_time_s": t + 0.5 + (0.0 if period == 1 else 2700.0),
                "player": f"{'Home' if team == 'H' else 'Away'}_{eid}",
                "team": "Home" if team == "H" else "Away",
                "start_x": ax,
                "start_y": ay,
                "end_x": ex,
                "end_y": ey,
            }
        )
    return pd.DataFrame(rows)


def _metrica_run(events: pd.DataFrame, home_starts_left: bool = True) -> pd.DataFrame:
    from silly_kicks.spadl import metrica

    return metrica.convert_to_actions(events, home_team_id="Home", home_team_start_left=home_starts_left)[0]


# ------------------------------------------------------------------------------------------------------------ sportec
def _sportec(home_starts_left: bool = True) -> pd.DataFrame:
    kinds = {
        "pass": ("Play", None),
        "cross": ("Play", "cross"),
        "shot": ("ShotAtGoal", None),
        "corner_crossed": ("Corner", "high"),
        "goalkick": ("GoalKick", None),
    }
    rows = []
    for eid, period, t, team, kind, x, y in SCENE:
        event_type, height = kinds[kind]
        ax, ay = _absolute(x, y, team, period, "per_period", home_starts_left)
        rows.append(
            {
                "match_id": "M1",
                "event_id": f"e{eid}",
                "event_type": event_type,
                "period": period,
                "timestamp_seconds": t,
                "player_id": f"P{eid}",
                "team": "T-HOME" if team == "H" else "T-AWAY",
                "x": ax,
                "y": ay,
                "play_height": height,
            }
        )
    return pd.DataFrame(rows)


def _sportec_run(events: pd.DataFrame, home_starts_left: bool = True) -> pd.DataFrame:
    from silly_kicks.spadl import sportec

    return sportec.convert_to_actions(events, home_team_id="T-HOME", home_team_start_left=home_starts_left)[0]


# ----------------------------------------------------------------------------------------------------- gradientsports
def _gradientsports(home_starts_left: bool = True, *, blocked: bool = False) -> pd.DataFrame:
    from silly_kicks.spadl import gradientsports as gs

    req = sorted(gs.EXPECTED_INPUT_COLUMNS)
    kinds = {"pass": ("PA", "O"), "cross": ("CR", "O"), "shot": ("SH", "O"), "corner_crossed": ("CR", "C")}
    kinds["goalkick"] = ("PA", "G")
    rows = []
    for eid, period, t, team, kind, x, y in SCENE:
        pe, sp = kinds[kind]
        ax, ay = _absolute(x, y, team, period, "per_period", home_starts_left)
        row: dict[str, Any] = {c: None for c in req}
        clock = t + (0.0 if period == 1 else 2700.0)
        row.update(
            {
                "game_id": 10502,
                "event_id": eid,
                "possession_event_id": eid,
                "period_id": period,
                "time_seconds": clock,
                "start_time": clock,
                "team_id": 100 if team == "H" else 200,
                "player_id": 1000 + eid,
                "game_event_type": "OTB",
                "possession_event_type": pe,
                "set_piece_type": sp,
                "ball_x": ax - L / 2,
                "ball_y": ay - W / 2,
                "body_type": "R",
            }
        )
        if pe in ("PA",):
            row.update({"pass_outcome_type": "C"})
        elif pe == "CR":
            row["cross_outcome_type"] = "B" if blocked and eid == BLOCKED_EID else "C"
        else:
            row["shot_outcome_type"] = "S"
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in ("possession_event_id", "player_id", "team_id"):
        df[col] = df[col].astype("Int64")
    for col in ("game_id", "event_id", "period_id"):
        df[col] = df[col].astype("int64")
    for col in ("time_seconds", "start_time", "ball_x", "ball_y"):
        df[col] = df[col].astype("float64")
    return df


def _gradientsports_run(events: pd.DataFrame, home_starts_left: bool = True) -> pd.DataFrame:
    from silly_kicks.spadl import gradientsports as gs

    return gs.convert_to_actions(
        events,
        home_team_id=100,
        home_team_start_left=home_starts_left,
        home_team_start_left_extratime=home_starts_left,
    )[0]


# -------------------------------------------------------------------------------------------------------- skillcorner
def _skillcorner(home_starts_left: bool = True) -> pd.DataFrame:
    ends = _ends()
    rows = []
    for eid, period, t, team, kind, x, y in SCENE:
        ex, ey = ends[eid]
        gi = {"corner_crossed": "corner_for", "goalkick": "goal_kick_for"}.get(kind)
        clock = t + (0.0 if period == 1 else 2700.0)
        rows.append(
            {
                "event_type": "player_possession",
                "event_id": f"{eid}_0",
                "period": period,
                "time_start": f"{int(clock // 60):02d}:{clock % 60:04.1f}",
                "team_id": "team_home" if team == "H" else "team_away",
                "player_id": f"p{eid}",
                "x_start": x - L / 2,
                "y_start": y - W / 2,
                "x_end": ex - L / 2,
                "y_end": ey - W / 2,
                "start_type": "pass_reception",
                "end_type": "shot" if kind == "shot" else "pass",
                "game_interruption_before": gi,
                "game_interruption_after": None,
                "is_header": False,
                "hand_pass": False,
                "player_targeted_x_reception": None,
                "player_targeted_y_reception": None,
                "player_targeted_third_pass": None,
                "player_targeted_channel_pass": None,
            }
        )
    return pd.DataFrame(rows)


def _skillcorner_run(events: pd.DataFrame, home_starts_left: bool = True) -> pd.DataFrame:
    from silly_kicks.spadl import skillcorner

    meta = json.loads((_DATASETS / "skillcorner" / "match_metadata.json").read_text())
    return skillcorner.convert_to_actions(events, meta)[0]


# ------------------------------------------------------------------------------------------------------------- kloppy
def _kloppy(home_starts_left: bool = True) -> Any:
    """The committed Sportec EventDataset, transformed to the converter's own frame, with its records replaced
    by the scene (copies of its own Pass/Shot events, so every field kloppy needs stays real)."""
    from kloppy import sportec as ksportec  # type: ignore[import-not-found]
    from kloppy.domain import (  # type: ignore[import-not-found]
        Orientation,
        PassQualifier,
        PassType,
        Point,
        SetPieceQualifier,
        SetPieceType,
    )

    from silly_kicks.spadl._kloppy_coordinates import socceraction_coordinate_system

    d = _DATASETS / "kloppy"
    raw = ksportec.load_event(event_data=str(d / "sportec_events.xml"), meta_data=str(d / "sportec_meta.xml"))
    ds = raw.transform(
        to_orientation=Orientation.HOME_AWAY, to_coordinate_system=socceraction_coordinate_system(raw.metadata)
    )
    tpl_pass = next(e for e in ds.records if type(e).__name__ == "PassEvent")
    tpl_shot = next(e for e in ds.records if type(e).__name__ == "ShotEvent")
    home, away = ds.metadata.teams[0], ds.metadata.teams[1]
    periods = {p.id: p for p in ds.metadata.periods}
    ends = _ends()
    quals = {
        "pass": [],
        "cross": [PassQualifier(value=PassType.CROSS)],
        "corner_crossed": [SetPieceQualifier(value=SetPieceType.CORNER_KICK), PassQualifier(value=PassType.HIGH_PASS)],
        "goalkick": [SetPieceQualifier(value=SetPieceType.GOAL_KICK)],
    }
    records = []
    for eid, period, t, team, kind, x, y in SCENE:
        side = home if team == "H" else away
        ax, ay = _absolute(x, y, team, period, "home_right", home_starts_left)
        ex, ey = _absolute(*ends[eid], team, period, "home_right", home_starts_left)
        common = {
            "event_id": f"e{eid}",
            "team": side,
            "player": side.players[eid % len(side.players)],
            "period": periods[period],
            "timestamp": dt.timedelta(seconds=t),
            "coordinates": Point(x=ax, y=ay),
            "ball_owning_team": side,
        }
        if kind == "shot":
            records.append(dataclasses.replace(tpl_shot, **common))
        else:
            records.append(
                dataclasses.replace(tpl_pass, **common, qualifiers=quals[kind], receiver_coordinates=Point(x=ex, y=ey))
            )
    out = copy.copy(ds)
    out.records = records  # type: ignore[attr-defined]
    return out


def _kloppy_run(dataset: Any, home_starts_left: bool = True) -> pd.DataFrame:
    from silly_kicks.spadl import kloppy as kloppy_mod

    return kloppy_mod.convert_to_actions(dataset, game_id="scene")[0]


def _reverse_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[::-1].reset_index(drop=True)


def _reverse_kloppy(dataset: Any) -> Any:
    out = copy.copy(dataset)
    out.records = list(reversed(list(dataset.records)))  # type: ignore[attr-defined]
    return out


@dataclass(frozen=True)
class DeliveryCase:
    build: Callable[..., Any]
    run: Callable[..., pd.DataFrame]
    reverse: Callable[[Any], Any]
    #: True when the provider's coordinates depend on ``home_starts_left`` (per-period absolute).
    per_period: bool


DELIVERY_CASES: dict[str, DeliveryCase] = {
    "statsbomb": DeliveryCase(_statsbomb, _statsbomb_run, _reverse_df, False),
    "opta": DeliveryCase(_opta, _opta_run, _reverse_df, False),
    "wyscout": DeliveryCase(_wyscout, _wyscout_run, _reverse_df, False),
    "metrica": DeliveryCase(_metrica, _metrica_run, _reverse_df, True),
    "sportec": DeliveryCase(_sportec, _sportec_run, _reverse_df, True),
    "gradientsports": DeliveryCase(_gradientsports, _gradientsports_run, _reverse_df, True),
    "skillcorner": DeliveryCase(_skillcorner, _skillcorner_run, _reverse_df, False),
    "kloppy": DeliveryCase(_kloppy, _kloppy_run, _reverse_kloppy, False),
}


def scene_rows(actions: pd.DataFrame) -> pd.DataFrame:
    """The converter's output rows that carry a scene event, keyed by scene event id (synthetic rows dropped)."""
    oid = actions["original_event_id"].astype("string")
    eid = oid.str.extract(r"^e?(\d+)", expand=False)
    keep = eid.notna() & ~actions.get("is_synthetic", pd.Series(False, index=actions.index)).fillna(False).astype(bool)
    keep &= actions["type_id"] != _dribble_id()
    out = actions.loc[keep].copy()
    out["scene_eid"] = eid[keep].astype(int).to_numpy()
    return out.drop_duplicates("scene_eid", keep="first").set_index("scene_eid").sort_index()


def _dribble_id() -> int:
    from silly_kicks.spadl import config as spadlconfig

    return spadlconfig.actiontype_id["dribble"]
