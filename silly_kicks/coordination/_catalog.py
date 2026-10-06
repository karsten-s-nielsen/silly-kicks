"""Coordination pair catalogue: which signal pairs are analysed at which level, by which methods (spec 7.7).

A ``PairSpec`` is a frozen, validated (level, signal_a, signal_b, role). The default catalogue is the spec 7.7
table; a caller may pass its own pairs. Vector coding runs only on same-unit (``commensurate``) pairs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from silly_kicks.coordination._columns import COORD_LEVELS, COORD_SIGNALS

_ALLOWED_ROLES: dict[str, set[str]] = {
    "team_team": {"canonical", "attacking_defending"},
    "cross_variable": {"attacking_defending"},
    "intra_team": {"same_team"},
    "dyad": {"same_team", "canonical"},
}


@dataclass(frozen=True)
class PairSpec:
    """A validated coordination pair: two signals, a level, and a pairing role.

    Examples
    --------
    >>> spec = PairSpec("team_team", "centroid_x", "centroid_x", "canonical")
    >>> (spec.axis, spec.commensurate)
    ('x', True)
    """

    level: str
    signal_a: str
    signal_b: str
    role: Literal["canonical", "attacking_defending", "same_team"]

    def __post_init__(self) -> None:
        if self.level not in COORD_LEVELS:
            raise ValueError(f"level must be one of {COORD_LEVELS}; got {self.level!r}")
        for sig in (self.signal_a, self.signal_b):
            if sig not in COORD_SIGNALS or sig == "possession":
                raise ValueError(f"signal {sig!r} is not an analysable COORD_SIGNALS entry")
        if self.role not in _ALLOWED_ROLES[self.level]:
            raise ValueError(f"role {self.role!r} is not allowed at level {self.level!r}")
        if self.level == "dyad":
            if not (self.signal_a == self.signal_b and self.signal_a in {"player_x", "player_y"}):
                raise ValueError("dyad pairs must be player_x==player_x or player_y==player_y")
        else:
            for sig in (self.signal_a, self.signal_b):
                if COORD_SIGNALS[sig].scope != "team":
                    raise ValueError(f"non-dyad level {self.level!r} requires team-scope signals; {sig!r} is not")

    @property
    def axis(self) -> str:
        """The pair's common signal axis, or ``"mixed"`` when the two axes differ (C7).

        Examples
        --------
        >>> PairSpec("team_team", "centroid_x", "centroid_y", "canonical").axis
        'mixed'
        """
        axis_a = COORD_SIGNALS[self.signal_a].axis
        axis_b = COORD_SIGNALS[self.signal_b].axis
        return axis_a if axis_a == axis_b else "mixed"

    @property
    def commensurate(self) -> bool:
        """True iff the two signals share a physical unit (the vector-coding requirement, spec 7.7).

        Examples
        --------
        >>> PairSpec("team_team", "convex_hull_area", "spread", "canonical").commensurate
        False
        """
        return COORD_SIGNALS[self.signal_a].unit == COORD_SIGNALS[self.signal_b].unit


_L1_CANONICAL = (
    "centroid_x",
    "centroid_y",
    "stretch_x",
    "stretch_y",
    "stretch_index",
    "spread",
    "convex_hull_area",
    "team_length",
    "team_width",
)

DEFAULT_PAIRS: tuple[PairSpec, ...] = (
    *(PairSpec("team_team", s, s, "canonical") for s in _L1_CANONICAL),
    PairSpec("cross_variable", "centroid_x", "defensive_line_x", "attacking_defending"),
    PairSpec("cross_variable", "team_length", "compactness_x", "attacking_defending"),
    PairSpec("cross_variable", "stretch_x", "stretch_x", "attacking_defending"),
    PairSpec("intra_team", "defensive_line_x", "centroid_x", "same_team"),
    PairSpec("dyad", "player_x", "player_x", "same_team"),
    PairSpec("dyad", "player_x", "player_x", "canonical"),
    PairSpec("dyad", "player_y", "player_y", "same_team"),
    PairSpec("dyad", "player_y", "player_y", "canonical"),
)

METHODS_BY_LEVEL: Mapping[str, tuple[str, ...]] = {
    "team_team": ("relative_phase", "cross_correlation", "vector_coding", "coherence"),
    "cross_variable": ("relative_phase", "cross_correlation", "vector_coding", "coherence"),
    "intra_team": ("relative_phase", "cross_correlation", "vector_coding", "coherence"),
    "dyad": ("relative_phase", "cross_correlation"),
}


def resolve_pairs(pairs: Sequence[PairSpec] | None, levels: Sequence[str]) -> tuple[PairSpec, ...]:
    """The pairs to analyse: the default catalogue (or the caller's) restricted to ``levels``.

    Examples
    --------
    >>> pairs = resolve_pairs(None, ["team_team"])
    >>> {p.level for p in pairs}
    {'team_team'}
    """
    chosen = DEFAULT_PAIRS if pairs is None else tuple(pairs)
    keep = set(levels)
    return tuple(p for p in chosen if p.level in keep)
