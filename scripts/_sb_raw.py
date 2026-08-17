"""StatsBomb raw-JSON parsers -- the single-sourced, scripts-side shaping layer.

``flatten_events`` is the ONE copy of a body that previously existed six times
(``build_sb360_coverage.py::_adapt_events``,
``build_worldcup_fixture.py::_adapt_events_to_silly_kicks_input`` and inline adapters in
``tests/spadl/test_add_possessions.py``, ``tests/spadl/test_cross_provider_parity.py``,
``tests/invariants/_loaders.py`` and ``tests/test_xthreat_statsbomb_e2e.py``). It maps raw
StatsBomb event dicts -- exactly the shape
``events.json`` carries, and the shape ``statsbombpy.sb.events(fmt="dict")`` returns -- to the
``silly_kicks.spadl.statsbomb`` converter's ``EXPECTED_INPUT_COLUMNS`` contract.

The three ``parse_*`` helpers shape the other pining ``statsbomb`` artifacts (freeze frames,
match metadata, roster) for the ``silly_kicks.providers.statsbomb`` port and the loader. They are
pure (payload in, plain data out) and ``.get()``-tolerant: a missing optional field yields a
documented default, never a ``KeyError`` mid-corpus.

This lives in ``scripts/`` deliberately: the ``providers/statsbomb`` port is shape-only and takes
already-loaded payloads (ADR-054), so raw-JSON parsing is a scripts-side concern and adds no
runtime dependency to the library.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

#: Raw event keys that map to dedicated SPADL-input columns; everything else on an event goes
#: into ``extra`` verbatim, where ``silly_kicks.spadl.statsbomb._flatten_extra`` reads it.
_TOP_LEVEL_KEYS = frozenset({"id", "period", "timestamp", "team", "player", "type", "location"})


def flatten_events(
    events: Sequence[dict],
    match_id: int,
    *,
    surface_native: Iterable[str] = (),
) -> pd.DataFrame:
    """Raw StatsBomb event dicts -> the ``silly_kicks.spadl.statsbomb`` input contract.

    Parameters
    ----------
    events
        Raw StatsBomb events for ONE match (``events.json`` decoded, or
        ``statsbombpy.sb.events(match_id, fmt="dict")``).
    match_id
        The match id, written to every row's ``game_id``.
    surface_native
        Raw top-level event keys to ALSO surface as dedicated columns, appended after ``extra``
        (they remain inside ``extra`` too). ``("possession",)`` reproduces the
        ``build_worldcup_fixture`` variant, whose caller then passes
        ``convert_to_actions(..., preserve_native=["possession"])``. Empty (the default)
        reproduces the ``build_sb360_coverage`` variant.
    """
    surface = tuple(surface_native)
    rows = []
    for e in events:
        row = {
            "game_id": match_id,
            "event_id": e.get("id"),
            "period_id": e.get("period"),
            "timestamp": e.get("timestamp"),
            "team_id": (e.get("team") or {}).get("id"),
            "player_id": (e.get("player") or {}).get("id"),
            "type_name": (e.get("type") or {}).get("name"),
            "location": e.get("location"),
            "extra": {k: v for k, v in e.items() if k not in _TOP_LEVEL_KEYS},
        }
        for key in surface:
            row[key] = e.get(key)
        rows.append(row)
    return pd.DataFrame(rows)


def parse_freeze_frames(raw: Sequence[dict] | dict) -> list[dict]:
    """Normalise a StatsBomb 360 payload to the ``shape_snapshots`` ``frames_raw`` contract.

    The open-data / licensed ``freeze_frames`` artifact is a list of records, each carrying
    ``event_uuid``, ``freeze_frame`` (the per-player array) and ``visible_area``. A dict wrapper
    (``{"frames": [...]}`` or similar) is unwrapped tolerantly. Returned verbatim -- the port
    (``silly_kicks.providers.statsbomb.shape_snapshots``) owns the actual shaping.
    """
    if isinstance(raw, dict):
        for key in ("frames", "freeze_frames", "three_sixty"):
            if isinstance(raw.get(key), list):
                return list(raw[key])
        return [raw]
    return list(raw)


def parse_metadata(raw: dict | Sequence[dict]) -> dict:
    """Extract ``home_team_id`` + fidelity versions from a StatsBomb match-metadata payload.

    Accepts a single match row (a dict) or a one-element list of match rows. ``home_team_id``
    is read from the nested ``home_team.home_team_id`` (falling back to a flat ``home_team_id``).
    Fidelity versions default to ``1`` when absent -- StatsBomb omits them on low-fidelity data --
    and are coerced to ``int`` because the JSON encodes them as strings (``"2"``).
    """
    if isinstance(raw, dict):
        row: dict = raw
    else:
        if len(raw) != 1:
            raise ValueError(
                f"parse_metadata expects a single match row; got a sequence of {len(raw)}. "
                "Select the row for this match before calling."
            )
        row = raw[0]

    home = row.get("home_team")
    if isinstance(home, dict):
        home_team_id = home.get("home_team_id", home.get("id"))
    else:
        home_team_id = row.get("home_team_id")

    def _fidelity(key: str) -> int:
        v = row.get(key)
        return 1 if v is None else int(v)

    return {
        "home_team_id": int(home_team_id) if home_team_id is not None else None,
        "xy_fidelity_version": _fidelity("xy_fidelity_version"),
        "shot_fidelity_version": _fidelity("shot_fidelity_version"),
    }


def parse_roster(raw: Sequence[dict]) -> dict[int, dict]:
    """StatsBomb lineups payload -> ``{player_id: {name, jersey, team, position}}``.

    The lineups artifact is a list of team dicts, each with ``team_id`` and a ``lineup`` list of
    player rows (``player_id``, ``player_name``, ``jersey_number``, and a ``positions`` list whose
    first entry carries the ``position`` name). Every field is read ``.get()``-tolerantly; a player
    with no recorded position gets ``position=None``.
    """
    out: dict[int, dict] = {}
    for team in raw or []:
        team_id = team.get("team_id", team.get("id"))
        for player in team.get("lineup", []) or []:
            pid = player.get("player_id", player.get("id"))
            if pid is None:
                continue
            positions = player.get("positions") or []
            position = positions[0].get("position") if positions else player.get("position")
            out[int(pid)] = {
                "name": player.get("player_name", player.get("name")),
                "jersey": player.get("jersey_number"),
                "team": int(team_id) if team_id is not None else None,
                "position": position,
            }
    return out
