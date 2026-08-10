"""Test-side helpers for the ADR-055 goal-map seam.

Why this module exists rather than a per-file ``resolve_defended_goals(frames)``: the re-key
touched 133 call sites, and at most of them ``home_team_id=H`` was not a statement about the
fixture's geometry at all -- it was a statement about the CONVENTION ("H defends x=0"). Deriving
the map from the fixture instead would silently re-decide that convention from wherever the
fixture happens to have parked its keeper, which turns a mechanical migration into 133 quiet
changes of what each test asserts.

So the migration has two spellings, and they are chosen deliberately per call site:

``goal_map_like_home_team_id(frames, H)``
    Byte-equivalent to the removed argument: every ``(game, period, team)`` present in ``frames``
    gets ``0.0`` if the team is ``H`` and ``105.0`` otherwise -- exactly what
    ``same_id(team_id, home_team_id)`` computed at the sites this cycle deleted. Use where the
    test is about something else and the orientation is scaffolding.

``resolve_defended_goals(frames)``
    The real seam. Use where the test is ABOUT geometry, orientation or the map itself, since
    that is the code path production takes.

The ``0.0``/``105.0`` fork below is a deliberate reproduction of the rule the package now owns in
one place. It lives in ``tests/`` on purpose: ``test_goal_map_population.py`` scans
``silly_kicks/`` only, and a test helper asserting *equivalence with the old behaviour* has to be
able to express the old behaviour.
"""

from __future__ import annotations

from types import MappingProxyType

import pandas as pd

from silly_kicks.id_compat import same_id
from silly_kicks.tracking import GoalMap

FIELD_LENGTH = 105.0


def goal_map_for(ends: dict, *, game_id: object = 1, period_id: object = 1, guessed: dict | None = None) -> GoalMap:
    """A ``GoalMap`` stating ``{team_id: defended_goal_x}`` for one ``(game, period)``.

    Keys are canonicalized through ``GoalMap._key``, because the mappings are keyed by canonical
    STRINGS -- a hand-built ``{(1, 1, 2): 105.0}`` would be a map every accessor misses.
    """
    resolved = {GoalMap._key(game_id, period_id, team): float(end) for team, end in ends.items()}
    guessed_keyed = {GoalMap._key(game_id, period_id, team): float(end) for team, end in (guessed or {}).items()}
    return GoalMap(MappingProxyType(resolved), MappingProxyType(guessed_keyed), frozenset())


def goal_map_like_home_team_id(frames: pd.DataFrame, home_team_id) -> GoalMap:
    """The map the deleted ``home_team_id`` argument implied, for every key in ``frames``.

    Covers every ``(game_id, period_id, team_id)`` actually present, so it works on multi-period
    and multi-game fixtures. Ball rows (NA team) are skipped -- they are in neither mapping,
    which is the seam's own ``unresolved`` treatment of an NA team identity.
    """
    resolved: dict[tuple, float] = {}
    cols = ["game_id", "period_id", "team_id"]
    for game, period, team in frames[cols].drop_duplicates().itertuples(index=False):
        if pd.isna(team):
            continue
        resolved[GoalMap._key(game, period, team)] = 0.0 if same_id(team, home_team_id) else FIELD_LENGTH
    return GoalMap(MappingProxyType(resolved), MappingProxyType({}), frozenset())
