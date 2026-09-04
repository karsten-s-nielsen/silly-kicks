"""Duel extraction for TF-55 (spec §5b.2): native sportec winner/loser + derived tackle/take_on adjacency.

The labeling strategy is chosen at FRAME-SET granularity (native if a populated ``tackle_winner_player_id``
is present, else derive), never per-duel-guess. Ground duels only (no SPADL aerial type). Indeterminate
duels (no clear winner) are EXCLUDED and counted (owner ruling; ADR-042). ids via ``id_compat`` (ADR-019).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import pandas as pd

from silly_kicks.id_compat import same_id
from silly_kicks.spadl import config as spadlconfig

_TACKLE = spadlconfig.actiontype_id["tackle"]
_TAKE_ON = spadlconfig.actiontype_id["take_on"]
_SUCCESS = spadlconfig.result_id["success"]
_ADJACENCY_WINDOW_S = 5.0  # a derived duel pairs a tackle + take_on within this many seconds


class DuelGame(NamedTuple):
    """One resolved duel: winner beats loser, in a game/period at a time. ``source`` in {native, derived}.

    Examples
    --------
    >>> from silly_kicks.duels import DuelGame
    >>> g = DuelGame(game_id=1, period_id=1, time_seconds=5.0, winner_player=100, winner_team=10,
    ...              loser_player=200, loser_team=20, source="native")
    >>> (g.winner_player, g.loser_player, g.source)
    (100, 200, 'native')
    """

    game_id: Any
    period_id: Any
    time_seconds: float
    winner_player: Any
    winner_team: Any
    loser_player: Any
    loser_team: Any
    source: str


@dataclass(frozen=True)
class DuelExtractReport:
    """Census over the candidate duels. ``n_native + n_derived + n_excluded == n_candidate``.

    Examples
    --------
    >>> from silly_kicks.duels import DuelExtractReport
    >>> r = DuelExtractReport(labeling_strategy="derived", n_candidate=16, n_native=0,
    ...                       n_derived=11, n_excluded=5)
    >>> r.n_native + r.n_derived + r.n_excluded == r.n_candidate
    True
    """

    labeling_strategy: str  # "native" or "derived"
    n_candidate: int
    n_native: int
    n_derived: int
    n_excluded: int


def _sorted(actions: pd.DataFrame) -> pd.DataFrame:
    return actions.sort_values(["game_id", "period_id", "time_seconds", "action_id"], kind="stable").reset_index(
        drop=True
    )


def _has_native(actions: pd.DataFrame) -> bool:
    return "tackle_winner_player_id" in actions.columns and bool(actions["tackle_winner_player_id"].notna().any())


def extract_duels(actions: pd.DataFrame) -> tuple[list[DuelGame], DuelExtractReport]:
    """Extract ground duels from SPADL ``actions``. Returns ``(games, DuelExtractReport)``.

    The strategy is chosen at frame-set granularity: native (sportec ``tackle_winner/loser``) if a
    populated winner column is present, else derived from the ``tackle`` / ``take_on`` adjacency.

    Examples
    --------
    Extract duels and read the census (``tests/duels/test_extract.py`` has the worked cases)::

        from silly_kicks.duels import extract_duels

        games, report = extract_duels(actions)
        # report.labeling_strategy is "native" or "derived"; games is a list of DuelGame
    """
    a = _sorted(actions)
    if _has_native(a):
        return _extract_native(a)
    return _extract_derived(a)


def _extract_native(a: pd.DataFrame) -> tuple[list[DuelGame], DuelExtractReport]:
    tackles = a[a["type_id"] == _TACKLE]
    games: list[DuelGame] = []
    n_candidate = n_excluded = 0
    for _, r in tackles.iterrows():
        winner, loser = r.get("tackle_winner_player_id"), r.get("tackle_loser_player_id")
        if pd.isna(winner) or pd.isna(loser):
            continue  # a tackle row without a native winner/loser is not a native duel
        n_candidate += 1
        games.append(
            DuelGame(
                r["game_id"],
                r["period_id"],
                float(r["time_seconds"]),
                winner,
                r.get("tackle_winner_team_id"),
                loser,
                r.get("tackle_loser_team_id"),
                "native",
            )
        )
    return games, DuelExtractReport("native", n_candidate, len(games), 0, n_excluded)


def _extract_derived(a: pd.DataFrame) -> tuple[list[DuelGame], DuelExtractReport]:
    # A derived duel = an adjacent (tackle, take_on) pair by opposing teams within the window. The winner
    # is whoever's action SUCCEEDED (tackle success -> tackler; take_on success -> dribbler); both-success
    # / both-fail is indeterminate -> excluded.
    games: list[DuelGame] = []
    n_candidate = n_excluded = 0
    tid = a["type_id"].to_numpy()
    for i in range(len(a) - 1):
        pair = {int(tid[i]), int(tid[i + 1])}
        if pair != {_TACKLE, _TAKE_ON}:
            continue
        r1, r2 = a.iloc[i], a.iloc[i + 1]
        if not same_id(r1["game_id"], r2["game_id"]) or same_id(r1["team_id"], r2["team_id"]):
            continue  # a duel is cross-team within one game
        if pd.isna(r1["player_id"]) or pd.isna(r2["player_id"]):
            continue  # a duel needs both participants identified
        if abs(float(r1["time_seconds"]) - float(r2["time_seconds"])) > _ADJACENCY_WINDOW_S:
            continue
        n_candidate += 1
        tackle, takeon = (r1, r2) if r1["type_id"] == _TACKLE else (r2, r1)
        t_win = tackle["result_id"] == _SUCCESS
        k_win = takeon["result_id"] == _SUCCESS
        if t_win and not k_win:
            winner, loser = tackle, takeon
        elif k_win and not t_win:
            winner, loser = takeon, tackle
        else:  # both succeeded or both failed -> no clear winner
            n_excluded += 1
            continue
        games.append(
            DuelGame(
                r1["game_id"],
                r1["period_id"],
                float(min(r1["time_seconds"], r2["time_seconds"])),
                winner["player_id"],
                winner["team_id"],
                loser["player_id"],
                loser["team_id"],
                "derived",
            )
        )
    return games, DuelExtractReport("derived", n_candidate, 0, len(games), n_excluded)
