"""Glicko-2 duel ratings (TF-55): the pure ``update_glicko`` primitive (one rating period).

Implements Glickman's Glicko-2 system verbatim (the SCALE=173.7178 transform, the Illinois volatility
iteration, and the inactivity RD-growth for a player who contests no duel this period). Pure -- no I/O,
no pandas. ``compute_duel_ratings`` (the resumable orchestrator over matches) is added in _compute below.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from typing import Any

import pandas as pd

from silly_kicks.id_compat import canonical_id, canonical_id_series

from ._columns import (
    DU_CONTESTED,
    DU_LOST,
    DU_RATING,
    DU_RATING_DEVIATION,
    DU_VOLATILITY,
    DU_WINNER_SOURCE,
    DU_WON,
    DUEL_COLUMNS,
)
from ._config import DuelRatingParams, GlickoState
from ._extract import extract_duels
from ._report import DuelRatingReport

_SCALE = 173.7178  # Glicko <-> Glicko-2 rating-scale factor
_DEFAULT = DuelRatingParams()
_WINDOW_STATS = frozenset({"as_of_end", "change"})  # trajectory-slice statistics (spec 5b.4)


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


def _expected(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def _new_volatility(delta: float, phi: float, v: float, sigma: float, tau: float) -> float:
    """The Glicko-2 volatility update via Illinois-algorithm root finding."""
    a = math.log(sigma * sigma)

    def f(x: float) -> float:
        ex = math.exp(x)
        num = ex * (delta * delta - phi * phi - v - ex)
        den = 2.0 * (phi * phi + v + ex) ** 2
        return num / den - (x - a) / (tau * tau)

    big_a = a
    if delta * delta > phi * phi + v:
        big_b = math.log(delta * delta - phi * phi - v)
    else:
        k = 1
        while f(a - k * tau) < 0.0:
            k += 1
        big_b = a - k * tau

    f_a, f_b = f(big_a), f(big_b)
    while abs(big_b - big_a) > 1e-6:
        c = big_a + (big_a - big_b) * f_a / (f_b - f_a)
        f_c = f(c)
        if f_c * f_b <= 0.0:
            big_a, f_a = big_b, f_b
        else:
            f_a /= 2.0
        big_b, f_b = c, f_c
    return math.exp(big_a / 2.0)


def _glicko2_update(
    state: GlickoState, opp_games: list[tuple[GlickoState, float]], params: DuelRatingParams
) -> GlickoState:
    mu = (state.rating - 1500.0) / _SCALE
    phi = state.rd / _SCALE
    sigma = state.volatility

    if not opp_games:  # inactive this period -> RD grows (uncertainty), rating + volatility unchanged
        if params.apply_inactivity_rd_growth:
            phi_star = math.sqrt(phi * phi + sigma * sigma)
            return GlickoState(state.rating, phi_star * _SCALE, sigma)
        return state

    v_inv = 0.0
    delta_sum = 0.0
    for opp, score in opp_games:
        mu_j = (opp.rating - 1500.0) / _SCALE
        phi_j = opp.rd / _SCALE
        g_j = _g(phi_j)
        e_j = _expected(mu, mu_j, phi_j)
        v_inv += g_j * g_j * e_j * (1.0 - e_j)
        delta_sum += g_j * (score - e_j)
    v = 1.0 / v_inv
    delta = v * delta_sum

    sigma_new = _new_volatility(delta, phi, v, sigma, params.tau)
    phi_star = math.sqrt(phi * phi + sigma_new * sigma_new)
    phi_new = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    mu_new = mu + phi_new * phi_new * delta_sum
    return GlickoState(_SCALE * mu_new + 1500.0, _SCALE * phi_new, sigma_new)


def update_glicko(
    ratings: Mapping[Any, GlickoState],
    period_games: Sequence[tuple[Any, Any, float]],
    *,
    params: DuelRatingParams = _DEFAULT,
) -> dict[Any, GlickoState]:
    """One Glicko-2 rating period. ``period_games`` is ``(player_a, player_b, score_a)`` (score_a in
    {1.0, 0.0}); returns the NEW ``{player -> GlickoState}`` for every player in ``ratings`` or a game
    (unseen players seeded from ``params``). Pure -- Glickman's worked example is the oracle.

    Examples
    --------
    Update the rated player in Glickman's Glicko-2 worked example::

        from silly_kicks.duels import DuelRatingParams, GlickoState, update_glicko

        ratings = {"P": GlickoState(1500, 200, 0.06), "A": GlickoState(1400, 30, 0.06),
                   "B": GlickoState(1550, 100, 0.06), "C": GlickoState(1700, 300, 0.06)}
        games = [("P", "A", 1.0), ("P", "B", 0.0), ("P", "C", 0.0)]
        new = update_glicko(ratings, games)  # new["P"] ~ (1464.06, 151.52, 0.05999)
    """
    games_by_player: dict[Any, list[tuple[Any, float]]] = defaultdict(list)
    for a, b, score_a in period_games:
        games_by_player[a].append((b, float(score_a)))
        games_by_player[b].append((a, 1.0 - float(score_a)))

    def state_of(p: Any) -> GlickoState:
        return ratings[p] if p in ratings else params.initial_state()

    players = set(ratings) | set(games_by_player)
    out: dict[Any, GlickoState] = {}
    for player in players:
        opp_games = [(state_of(opp), s) for opp, s in games_by_player.get(player, [])]
        out[player] = _glicko2_update(state_of(player), opp_games, params)
    return out


def compute_duel_ratings(
    actions: pd.DataFrame,
    *,
    initial_ratings: Mapping[Any, GlickoState] | None = None,
    window: Collection[Any] | None = None,
    window_stat: str = "as_of_end",
    params: DuelRatingParams = _DEFAULT,
) -> tuple[pd.DataFrame, DuelRatingReport]:
    """Per-(player, match) Glicko-2 duel ratings from SPADL ``actions``.

    Extracts duels (native sportec winner/loser, else the tackle/take_on adjacency), then processes
    each MATCH as one Glicko-2 rating period in ascending-``game_id`` order (a chronology proxy when
    no match date is available), carrying ratings forward across matches. Emits one row per
    ``(game_id, player_id)`` for every player who CONTESTED >= 1 duel that match, holding the player's
    post-match ``(rating, RD, volatility)`` and win/loss tallies. Player-id keys are canonicalised
    (ADR-019); the raw ``player_id`` is emitted. Indeterminate duels are excluded and counted (ADR-042).

    ``initial_ratings`` (keyed by any id representation -- canonicalised on ingest) SEEDS the running
    state so a later batch RESUMES rather than recomputing the world; the running state after the last
    match is returned as ``DuelRatingReport.final_ratings`` (canonical-keyed), so
    ``compute_duel_ratings(m1+m2)`` final ratings equal two batches threaded via ``initial_ratings``.

    ``window`` (a collection of ``game_id``s) slices the trajectory (spec 5b.4) -- one row per player
    (``game_id`` NA), ratings are cumulative NOT summed: ``window_stat="as_of_end"`` (default) is the
    player's latest rating within the window; ``window_stat="change"`` is its change from just before
    the window (the pre-window state, or the seed for a window debutant). Counts are summed.

    Returns ``(samples, DuelRatingReport)`` with columns/dtypes pinned by ``DUEL_COLUMNS``.

    Examples
    --------
    Rate one match's duels (see ``tests/duels/`` for runnable coverage)::

        from silly_kicks.duels import compute_duel_ratings

        samples, report = compute_duel_ratings(actions)
        # samples: one row per (game_id, player_id) with duel_rating / _deviation / _volatility
    """
    if window_stat not in _WINDOW_STATS:
        raise ValueError(f"unknown window_stat {window_stat!r}; expected one of {sorted(_WINDOW_STATS)}")

    games, extract_report = extract_duels(actions)

    by_game: dict[Any, list[Any]] = defaultdict(list)
    for d in games:
        by_game[d.game_id].append(d)
    try:
        ordered = sorted(by_game)
    except TypeError:  # heterogeneous game_id types -> stable string order
        ordered = sorted(by_game, key=str)
    game_order = {g: i for i, g in enumerate(ordered)}

    seed_map = {canonical_id(k): v for k, v in initial_ratings.items()} if initial_ratings else {}
    ratings: dict[Any, GlickoState] = dict(seed_map)
    rows: list[dict[str, Any]] = []
    for game_id in ordered:
        period_games: list[tuple[Any, Any, float]] = []
        tally: dict[Any, dict[str, Any]] = {}
        for d in by_game[game_id]:
            w, loser = canonical_id(d.winner_player), canonical_id(d.loser_player)
            period_games.append((w, loser, 1.0))
            tw = tally.setdefault(w, {"contested": 0, "won": 0, "lost": 0, "raw": d.winner_player, "source": d.source})
            tw["contested"] += 1
            tw["won"] += 1
            tl = tally.setdefault(
                loser, {"contested": 0, "won": 0, "lost": 0, "raw": d.loser_player, "source": d.source}
            )
            tl["contested"] += 1
            tl["lost"] += 1
        ratings = update_glicko(ratings, period_games, params=params)
        for canon_p, t in tally.items():
            st = ratings[canon_p]
            rows.append(
                {
                    "game_id": game_id,
                    "player_id": t["raw"],
                    DU_RATING: st.rating,
                    DU_RATING_DEVIATION: st.rd,
                    DU_VOLATILITY: st.volatility,
                    DU_CONTESTED: t["contested"],
                    DU_WON: t["won"],
                    DU_LOST: t["lost"],
                    DU_WINNER_SOURCE: t["source"],
                    "_canon": canon_p,
                    "_order": game_order[game_id],
                }
            )

    n_player_match_rows = len(rows)
    if window is not None:
        rows = _slice_window(rows, window, window_stat, seed_map, params)

    samples = pd.DataFrame(rows).reindex(columns=list(DUEL_COLUMNS)).astype(DUEL_COLUMNS)
    report = DuelRatingReport(
        params=params,
        labeling_strategy=extract_report.labeling_strategy,
        n_matches=len(ordered),
        n_duels=len(games),
        n_duels_excluded=extract_report.n_excluded,
        n_player_match_rows=n_player_match_rows,
        final_ratings=ratings,
    )
    return samples, report


def _slice_window(
    rows: list[dict[str, Any]],
    window: Collection[Any],
    window_stat: str,
    seed_map: Mapping[Any, GlickoState],
    params: DuelRatingParams,
) -> list[dict[str, Any]]:
    """Collapse the per-(player, match) snapshots to one trajectory-slice row per player (spec 5b.4)."""
    wanted = set(canonical_id_series(pd.Series(list(window), dtype="object")))
    full: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        full[r["_canon"]].append(r)
    out: list[dict[str, Any]] = []
    for canon_p, frows in full.items():
        frows.sort(key=lambda r: r["_order"])
        win = [r for r in frows if canonical_id(r["game_id"]) in wanted]
        if not win:
            continue
        last = win[-1]
        contested = sum(r[DU_CONTESTED] for r in win)
        won = sum(r[DU_WON] for r in win)
        lost = sum(r[DU_LOST] for r in win)
        if window_stat == "as_of_end":
            rating, rd, vol = last[DU_RATING], last[DU_RATING_DEVIATION], last[DU_VOLATILITY]
        else:  # "change": last-in-window minus the state just BEFORE the window (pre-window row, else seed)
            pre = [r for r in frows if r["_order"] < win[0]["_order"]]
            base = pre[-1] if pre else None
            if base is not None:
                base_rating, base_rd, base_vol = base[DU_RATING], base[DU_RATING_DEVIATION], base[DU_VOLATILITY]
            else:
                seed = seed_map.get(canon_p) or params.initial_state()
                base_rating, base_rd, base_vol = seed.rating, seed.rd, seed.volatility
            rating = last[DU_RATING] - base_rating
            rd = last[DU_RATING_DEVIATION] - base_rd
            vol = last[DU_VOLATILITY] - base_vol
        out.append(
            {
                "game_id": pd.NA,
                "player_id": last["player_id"],
                DU_RATING: rating,
                DU_RATING_DEVIATION: rd,
                DU_VOLATILITY: vol,
                DU_CONTESTED: contested,
                DU_WON: won,
                DU_LOST: lost,
                DU_WINNER_SOURCE: last[DU_WINNER_SOURCE],
            }
        )
    return out
