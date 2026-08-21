"""Runnable per-converter cases for the order-insensitivity gate.

Each SPADL ``convert_to_actions`` in ``silly_kicks/spadl/`` gets ONE non-e2e
:class:`ConverterCase`: a native input (that converter's
``EXPECTED_INPUT_COLUMNS`` / native schema), the ``convert_to_actions`` call,
and a deterministic ``permute`` that reverses the input's timestamp-blocks while
preserving within-timestamp order. The gate in
``test_converter_order_insensitivity.py`` runs the converter on the input and on
its permutation and asserts content-equality (drop ``action_id``), plus a
per-``(game_id, period_id)`` index-chronology check.

Fixture provenance per case is recorded in ``ConverterCase.fixture_kind``
(``"committed"`` reuses a committed fixture; ``"synthetic"`` builds a small
multi-timestamp *non-chronological* input, so both the permutation and the
index-chronology gate are non-vacuous -- a gate that only sees pre-sorted input
proves nothing).
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_DATASETS = Path(__file__).parent.parent / "datasets"


# ---------------------------------------------------------------------------
# Generic permutation primitives
# ---------------------------------------------------------------------------
def _reverse_timestamp_blocks(df: pd.DataFrame, block_cols: tuple[str, ...]) -> pd.DataFrame:
    """Reverse the order of the timestamp-blocks, preserving within-block order.

    A "block" is a maximal set of rows sharing the same ``block_cols`` value.
    Blocks are ordered by first appearance, then that order is reversed; rows
    within a block keep their original relative order (R3: never disturb
    genuinely-ambiguous intra-timestamp ties). Deterministic and guaranteed
    non-identity for >1 distinct block. Never mutates the input.
    """
    df = df.reset_index(drop=True)
    if len(df) == 0:
        return df.copy()
    blk = df.groupby(list(block_cols), dropna=False, sort=False).ngroup().to_numpy()
    rev_rank = blk.max() - blk
    within = np.arange(len(df))
    # np.lexsort: last key is primary -> order by reversed-block-rank, then original position.
    order = np.lexsort((within, rev_rank))
    return df.iloc[order].reset_index(drop=True)


def _reverse_blocks_list(items: list, key: Callable[[Any], Any]) -> list:
    """List analogue of :func:`_reverse_timestamp_blocks` for kloppy events."""
    keyed = [str(key(it)) for it in items]
    order: list[str] = []
    seen: set[str] = set()
    for k in keyed:
        if k not in seen:
            seen.add(k)
            order.append(k)
    rank = {k: i for i, k in enumerate(order)}
    n = len(order)
    idx = sorted(range(len(items)), key=lambda i: (n - 1 - rank[keyed[i]], i))
    return [items[i] for i in idx]


def _df_signature(df: pd.DataFrame) -> tuple:
    """Row-order signature of a native-input frame (stringified, order-sensitive)."""
    return tuple(map(tuple, df.reset_index(drop=True).astype(str).to_numpy().tolist()))


@dataclass(frozen=True)
class ConverterCase:
    """One runnable, permutable case for a single ``convert_to_actions``."""

    name: str
    build_input: Callable[[], Any]
    run: Callable[[Any], pd.DataFrame]
    permute: Callable[[Any], Any]
    signature: Callable[[Any], Any]
    fixture_kind: str  # "committed" | "synthetic"
    sort_key_columns: tuple[str, ...]


# ---------------------------------------------------------------------------
# statsbomb -- synthetic, multi-timestamp, non-chronological
# ---------------------------------------------------------------------------
def _statsbomb_input() -> pd.DataFrame:
    def _p(eid, ts, player, sx, ex):
        return {
            "game_id": 1,
            "event_id": eid,
            "period_id": 1,
            "timestamp": ts,
            "team_id": 100,
            "player_id": player,
            "type_name": "Pass",
            "location": [sx, 40.0],
            "extra": {
                "pass": {
                    "end_location": [ex, 40.0],
                    "outcome": {"name": "Complete"},
                    "height": {"name": "Ground Pass"},
                }
            },
        }

    # File order is deliberately NOT chronological (20, 40, 10, 30 s).
    return pd.DataFrame(
        [
            _p("e2", "00:00:20.000", 202, 40.0, 50.0),
            _p("e4", "00:00:40.000", 204, 80.0, 90.0),
            _p("e1", "00:00:10.000", 201, 20.0, 30.0),
            _p("e3", "00:00:30.000", 203, 60.0, 70.0),
        ]
    )


def _statsbomb_run(events: pd.DataFrame) -> pd.DataFrame:
    from silly_kicks.spadl import statsbomb

    actions, _ = statsbomb.convert_to_actions(events, home_team_id=100)
    return actions


# ---------------------------------------------------------------------------
# opta -- synthetic, multi-timestamp, non-chronological. Deliberately exercises
# BOTH of opta's order-dependent PRE-SORT ``.shift()`` paths (they run before the
# time-sort at opta.py:203, so a plain-pass input gives the gate zero signal on
# them):
#   * ``_fix_recoveries`` (opta.py:516) resolves a "ball recovery" via a
#     file-adjacent ``.shift(-1)`` -> dribble vs non_action by neighbour distance.
#   * ``_fix_unintentional_ball_touches`` (opta.py:576) resolves a deflected
#     pass's end via ``.shift(-2)`` when the file-next event is a "ball touch".
# The events are positioned so the shift-target differs between file order and
# the reversed-timestamp-block order.
# ---------------------------------------------------------------------------
def _opta_input() -> pd.DataFrame:
    def _ev(eid, sec, player, type_name, sx, sy, ex, ey, outcome=True):
        return {
            "game_id": 1,
            "event_id": eid,
            "type_id": 1,
            "period_id": 1,
            "minute": 0,
            "second": sec,
            "team_id": 10,
            "player_id": player,
            "outcome": outcome,
            "start_x": sx,
            "start_y": sy,
            "end_x": ex,
            "end_y": ey,
            "qualifiers": {124: True},
            "type_name": type_name,
        }

    # File order (positions 0..6); distinct seconds => reversal fully reverses it.
    #   R's file-next is A_near (co-located -> non_action, dropped); reversed, R's
    #   next is A_far (distant -> dribble, survives) => the recovery's fate flips.
    #   P's file-next is the deflecting "ball touch" (outcome=True) -> P.end takes
    #   Q's start; reversed, P's next is a pass -> P keeps its own end.
    return pd.DataFrame(
        [
            _ev(70, 29, 207, "pass", 90.0, 10.0, 90.0, 10.0),  # A_far
            _ev(60, 31, 206, "ball recovery", 50.0, 34.0, 50.0, 34.0),  # R
            _ev(50, 30, 205, "pass", 50.0, 34.0, 55.0, 34.0),  # A_near
            _ev(20, 20, 202, "pass", 40.0, 34.0, 45.0, 34.0),  # P (deflected pass)
            _ev(21, 21, 203, "ball touch", 46.0, 34.0, 46.0, 34.0, outcome=True),  # deflection
            _ev(22, 22, 204, "pass", 99.0, 10.0, 99.0, 10.0),  # Q (P's shift(-2) target)
            _ev(10, 10, 201, "pass", 20.0, 34.0, 30.0, 34.0),  # filler
        ]
    )


def _opta_run(events: pd.DataFrame) -> pd.DataFrame:
    from silly_kicks.spadl import opta

    actions, _ = opta.convert_to_actions(events, home_team_id=10)
    return actions


# ---------------------------------------------------------------------------
# wyscout -- synthetic, multi-timestamp, non-chronological.
# Includes a duel->simulation pair (exercises the ``_fix_wyscout_events`` shift
# logic) alongside plain passes, so the permutation gate is not vacuous over the
# order-dependent event->action conversion, not merely the index numbering.
# ---------------------------------------------------------------------------
def _wyscout_input() -> pd.DataFrame:
    def _pass(eid, ms, player, sx, ex):
        return {
            "type_id": 8,
            "subtype_name": "Simple pass",
            "subtype_id": 85,
            "tags": [{"id": 1801}],
            "player_id": player,
            "positions": [{"y": 50, "x": sx}, {"y": 50, "x": ex}],
            "game_id": 1,
            "type_name": "Pass",
            "team_id": 100,
            "period_id": 1,
            "milliseconds": ms,
            "event_id": eid,
        }

    duel = {
        "type_id": 1,
        "subtype_name": "Ground attacking duel",
        "subtype_id": 11,
        "tags": [{"id": 503}, {"id": 701}, {"id": 1802}],
        "player_id": 301,
        "positions": [{"y": 48, "x": 82}, {"y": 47, "x": 83}],
        "game_id": 1,
        "type_name": "Duel",
        "team_id": 100,
        "period_id": 1,
        "milliseconds": 25000.0,
        "event_id": 5,
    }
    simulation = {
        "type_id": 2,
        "subtype_name": "Simulation",
        "subtype_id": 25,
        "tags": [{"id": 1702}],
        "player_id": 301,
        "positions": [{"y": 47, "x": 83}, {"y": 0, "x": 0}],
        "game_id": 1,
        "type_name": "Foul",
        "team_id": 100,
        "period_id": 1,
        "milliseconds": 27000.0,
        "event_id": 6,
    }
    # File order is deliberately NOT chronological.
    return pd.DataFrame(
        [
            _pass(2, 20000.0, 202, 40, 50),
            simulation,  # 27 s
            _pass(4, 40000.0, 204, 80, 90),
            duel,  # 25 s
            _pass(1, 10000.0, 201, 20, 30),
            _pass(3, 30000.0, 203, 60, 70),
        ]
    )


def _wyscout_run(events: pd.DataFrame) -> pd.DataFrame:
    from silly_kicks.spadl import wyscout

    actions, _ = wyscout.convert_to_actions(events, home_team_id=100)
    return actions


# ---------------------------------------------------------------------------
# sportec -- committed real IDSSE 9-event native slice. Its raw event order
# carries a genuine time inversion (..., 28.806, 26.370), so it is
# non-chronological by construction (spec exec-summary).
# ---------------------------------------------------------------------------
def _sportec_input() -> pd.DataFrame:
    return pd.read_parquet(_DATASETS / "sportec" / "idsse_slice" / "idsse_events_native_golden.parquet")


def _sportec_run(events: pd.DataFrame) -> pd.DataFrame:
    from silly_kicks.spadl import sportec

    actions, _ = sportec.convert_to_actions(events, home_team_id="home", home_team_start_left=True)
    return actions


# ---------------------------------------------------------------------------
# metrica -- synthetic, multi-timestamp, non-chronological, PLACEHOLDER ends
# (end == start) so ``_derive_end_coordinates`` fires and its output is
# order-dependent on the real (distinct-event-id) rows.
# ---------------------------------------------------------------------------
def _metrica_input() -> pd.DataFrame:
    def _p(eid, t, player, x):
        return {
            "match_id": "G1",
            "event_id": eid,
            "type": "PASS",
            "subtype": None,
            "period": 1,
            "start_time_s": t,
            "end_time_s": t + 0.5,
            "player": player,
            "team": "Home",
            "start_x": x,
            "start_y": 34.0,
            "end_x": x,  # placeholder end (== start) -> derive fires
            "end_y": 34.0,
        }

    return pd.DataFrame(
        [
            _p("e2", 20.0, "Home_2", 40.0),
            _p("e4", 40.0, "Home_4", 80.0),
            _p("e1", 10.0, "Home_1", 20.0),
            _p("e3", 30.0, "Home_3", 60.0),
        ]
    )


def _metrica_run(events: pd.DataFrame) -> pd.DataFrame:
    from silly_kicks.spadl import metrica

    actions, _ = metrica.convert_to_actions(events, home_team_id="Home", home_team_start_left=True)
    return actions


# ---------------------------------------------------------------------------
# skillcorner -- committed real possession fixture (15 events, 15 timestamps).
# ---------------------------------------------------------------------------
def _skillcorner_input() -> pd.DataFrame:
    return pd.read_csv(_DATASETS / "skillcorner" / "basic_possessions.csv")


def _skillcorner_run(events: pd.DataFrame) -> pd.DataFrame:
    from silly_kicks.spadl import skillcorner

    with open(_DATASETS / "skillcorner" / "match_metadata.json") as f:
        meta = json.load(f)
    actions, _ = skillcorner.convert_to_actions(events, meta)
    return actions


# ---------------------------------------------------------------------------
# gradientsports -- synthetic, multi-timestamp, non-chronological.
# GS initializes end == start for every event, so ``_derive_end_coordinates``
# fires for pass-class rows and its output is order-dependent.
# ---------------------------------------------------------------------------
def _gradientsports_input() -> pd.DataFrame:
    from silly_kicks.spadl import gradientsports as gs

    req = sorted(gs.EXPECTED_INPUT_COLUMNS)

    def _p(eid, t, player, x):
        base: dict[str, Any] = {c: None for c in req}
        base.update(
            {
                "game_id": 10502,
                "event_id": eid,
                "possession_event_id": eid,
                "period_id": 1,
                "time_seconds": t,
                # start_time == game clock here (monotonic in intended chronological order); it is
                # the order-insensitive basis for imputing the NaN-time FOUL below (Option D).
                "start_time": t,
                "team_id": 100,
                "player_id": player,
                "game_event_type": "OTB",
                "possession_event_type": "PA",
                "set_piece_type": "O",
                "ball_x": x,
                "ball_y": 0.0,
                "pass_outcome_type": "C",
                "body_type": "R",
            }
        )
        return base

    def _foul(eid, start_time, player, x):
        # Dedicated FOUL: NULL startGameClock -> NaN time_seconds, imputed by start_time-ordered
        # ffill (Option D). Its presence makes this case exercise the order-insensitive foul path.
        base: dict[str, Any] = {c: None for c in req}
        base.update(
            {
                "game_id": 10502,
                "event_id": eid,
                "possession_event_id": eid,
                "period_id": 1,
                "time_seconds": float("nan"),
                "start_time": start_time,
                "team_id": 100,
                "player_id": player,
                "game_event_type": "FOUL",
                "possession_event_type": "FO",
                "set_piece_type": "O",
                "ball_x": x,
                "ball_y": 0.0,
                "body_type": "R",
                "foul_type": "I",
                "final_foul_outcome_type": "Y",
            }
        )
        return base

    df = pd.DataFrame(
        [
            _p(2, 20.0, 2, -30.0),
            _p(4, 40.0, 4, 30.0),
            _foul(5, 25.0, 1, 5.0),  # NaN time_seconds; start_time 25 -> imputes to pass@20
            _p(1, 10.0, 1, -10.0),
            _p(3, 30.0, 3, 10.0),
        ]
    )
    for col in (
        "possession_event_id",
        "player_id",
        "team_id",
        "carry_defender_player_id",
        "challenger_player_id",
        "challenger_team_id",
        "challenge_winner_player_id",
        "challenge_winner_team_id",
    ):
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    df["game_id"] = df["game_id"].astype("int64")
    df["event_id"] = df["event_id"].astype("int64")
    df["period_id"] = df["period_id"].astype("int64")
    df["time_seconds"] = df["time_seconds"].astype("float64")
    df["start_time"] = df["start_time"].astype("float64")
    df["ball_x"] = df["ball_x"].astype("float64")
    df["ball_y"] = df["ball_y"].astype("float64")
    return df


def _gradientsports_run(events: pd.DataFrame) -> pd.DataFrame:
    from silly_kicks.spadl import gradientsports as gs

    actions, _ = gs.convert_to_actions(
        events,
        home_team_id=100,
        home_team_start_left=True,
        home_team_start_left_extratime=True,
    )
    return actions


# ---------------------------------------------------------------------------
# kloppy -- committed real Sportec EventDataset (parsed via kloppy, installed +
# fixture committed, so non-e2e). Input is an EventDataset, permuted by
# reordering its records; signature is the ordered event-id sequence.
# ---------------------------------------------------------------------------
def _kloppy_input() -> Any:
    from kloppy import sportec as ksportec  # type: ignore[import-not-found]

    d = _DATASETS / "kloppy"
    return ksportec.load_event(
        event_data=str(d / "sportec_events.xml"),
        meta_data=str(d / "sportec_meta.xml"),
    )


def _kloppy_run(dataset: Any) -> pd.DataFrame:
    from silly_kicks.spadl import kloppy as kloppy_mod

    actions, _ = kloppy_mod.convert_to_actions(dataset, game_id="order_gate")
    return actions


def _kloppy_permute(dataset: Any) -> Any:
    records = list(dataset.records)
    reordered = _reverse_blocks_list(records, key=lambda ev: (ev.period.id, ev.timestamp.total_seconds()))
    new = copy.copy(dataset)
    new.records = reordered  # type: ignore[attr-defined]
    return new


def _kloppy_signature(dataset: Any) -> tuple:
    return tuple(ev.event_id for ev in dataset.events)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def _df_case(name, build_input, run, block_cols, fixture_kind) -> ConverterCase:
    return ConverterCase(
        name=name,
        build_input=build_input,
        run=run,
        permute=lambda df, _bc=block_cols: _reverse_timestamp_blocks(df, _bc),
        signature=_df_signature,
        fixture_kind=fixture_kind,
        sort_key_columns=block_cols,
    )


CONVERTER_CASES: dict[str, ConverterCase] = {
    "statsbomb": _df_case("statsbomb", _statsbomb_input, _statsbomb_run, ("period_id", "timestamp"), "synthetic"),
    "opta": _df_case("opta", _opta_input, _opta_run, ("period_id", "minute", "second"), "synthetic"),
    "wyscout": _df_case("wyscout", _wyscout_input, _wyscout_run, ("period_id", "milliseconds"), "synthetic"),
    "sportec": _df_case("sportec", _sportec_input, _sportec_run, ("period", "timestamp_seconds"), "committed"),
    "metrica": _df_case("metrica", _metrica_input, _metrica_run, ("period", "start_time_s"), "synthetic"),
    "skillcorner": _df_case("skillcorner", _skillcorner_input, _skillcorner_run, ("period", "time_start"), "committed"),
    "gradientsports": _df_case(
        "gradientsports", _gradientsports_input, _gradientsports_run, ("period_id", "time_seconds"), "synthetic"
    ),
    "kloppy": ConverterCase(
        name="kloppy",
        build_input=_kloppy_input,
        run=_kloppy_run,
        permute=_kloppy_permute,
        signature=_kloppy_signature,
        fixture_kind="committed",
        sort_key_columns=("period_id", "time_seconds"),
    ),
}


def discover_converter_modules() -> set[str]:
    """Every ``silly_kicks/spadl/*.py`` module that defines a top-level
    ``convert_to_actions`` -- the surface the gate must cover (anti-rot meta)."""
    import ast

    import silly_kicks.spadl as _spadl_pkg

    spadl_dir = Path(_spadl_pkg.__file__).parent
    found: set[str] = set()
    for py in sorted(spadl_dir.glob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "convert_to_actions":
                found.add(py.stem)
    return found
