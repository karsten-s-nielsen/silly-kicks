"""Audit report for compute_duel_ratings (TF-55). Conserves the duel census (ADR-042)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._config import DuelRatingParams, GlickoState


@dataclass(frozen=True)
class DuelRatingReport:
    """Census over one ``compute_duel_ratings`` call.

    ``labeling_strategy`` is native (sportec winner/loser) or derived (tackle/take_on adjacency),
    chosen at frame-set granularity. ``n_duels`` is the scored (winner-resolved) duels;
    ``n_duels_excluded`` the indeterminate ones (no clear winner). Each scored duel contributes to
    exactly two player-match tallies, deduped per (player, match) into ``n_player_match_rows``.
    ``final_ratings`` is the running Glicko-2 state after the last match, keyed on the CANONICAL
    player id -- pass it straight back as ``initial_ratings`` to resume a later batch.

    Examples
    --------
    >>> from silly_kicks.duels import DuelRatingParams
    >>> from silly_kicks.duels._report import DuelRatingReport
    >>> r = DuelRatingReport(DuelRatingParams(), labeling_strategy="native", n_matches=1,
    ...                      n_duels=3, n_duels_excluded=1, n_player_match_rows=4)
    >>> r.n_duels + r.n_duels_excluded  # candidate duels this call
    4
    """

    params: DuelRatingParams
    labeling_strategy: str
    n_matches: int
    n_duels: int
    n_duels_excluded: int
    n_player_match_rows: int
    final_ratings: dict[Any, GlickoState] = field(default_factory=dict)
