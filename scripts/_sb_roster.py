"""Driver-side StatsBomb roster -> ``{team_id: gk_id}`` helper (ADR-054).

The library never parses raw provider JSON: the ``providers/statsbomb`` port is pure-shaping and
the keeper-identity resolver takes an already-built ``{team_id: gk_id}`` roster. This helper builds
that roster from a parsed StatsBomb lineup on the DRIVER side, so the raw-parse dependency stays out
of ``silly_kicks``.

Pure: no I/O, no ``statsbombpy`` import. A single dict-in, dict-out transform.
"""

from __future__ import annotations


def build_gk_roster_map(roster: dict[int, dict]) -> dict[object, object]:
    """Turn a parsed roster into the ``{team_id: goalkeeper_player_id}`` map the resolver consumes.

    Parameters
    ----------
    roster : dict[int, dict]
        ``parse_roster(...)`` output: ``{player_id: {"name", "jersey", "team", "position"}}``.

    Returns
    -------
    dict[object, object]
        ``{team_id: gk_id}`` for every team whose lineup names a ``"Goalkeeper"``. If a team lists
        MORE than one goalkeeper (a named substitute), the FIRST encountered is kept -- the goal-kick
        actor-event rung of ``resolve_keeper_identities`` resolves an in-match keeper substitution at
        resolution time, so nothing extra is recorded here.
    """
    gk_by_team: dict[object, object] = {}
    for player_id, info in roster.items():
        if info.get("position") != "Goalkeeper":
            continue
        team = info.get("team")
        if team not in gk_by_team:  # first goalkeeper per team wins (a named sub is left to the event rung)
            gk_by_team[team] = player_id
    return gk_by_team
