"""StatsBomb OPEN-DATA loader for the TF-54b bundled model + validation (public, redistributable).

The pining ``statsbomb`` provider is a private women's-soccer corpus; the bundled pass-completion
model and the construct-validity battery want the PUBLIC men's FIFA World Cup 2022 open data (the
corpus the spec + the ``@e2e`` test use, and the corpus the locked elite-defender prior matches).
StatsBomb open data is redistributable (github.com/statsbomb/open-data), so a model trained on it is
publicly reproducible.

Yields the same ``(provider, match_id, actions, frames, home_team_id)`` 5-tuple as
``scripts._loader_pining.load_matches`` so the ``for_each`` drivers consume it unchanged. Event-only:
``frames`` is an empty DataFrame (the counterfactual metric + the completion model never read frames).
``player_name`` is attached from the raw events (each carries ``player.{id, name}``) so the elite-prior
name resolution has real names to match, mirroring the pining path's roster join.

``statsbombpy`` is an optional ``scripts/`` dependency (network-gated); import is function-local.
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

#: FIFA World Cup 2022 (male) -- the spec's corpus; the locked elite-defender prior matches it.
WORLD_CUP_2022 = (43, 106)


def _values(payload) -> list:
    """statsbombpy returns a dict-keyed-by-id (``fmt="dict"``) or a list depending on version.

    The untyped ``payload`` is deliberate: the return type depends on the runtime ``fmt`` string,
    which statsbombpy does not model in-type -- mirrors ``scripts/build_sb360_coverage.py::_values``.
    """
    return list(payload.values()) if isinstance(payload, dict) else list(payload)


def _player_id_to_name(events: list[dict]) -> dict[int, str]:
    """``player_id -> player_name`` from the raw StatsBomb events (each event carries ``player``)."""
    out: dict[int, str] = {}
    for e in events:
        p = e.get("player")
        if isinstance(p, dict) and p.get("id") is not None:
            out[int(p["id"])] = str(p.get("name")) if p.get("name") is not None else None  # type: ignore[assignment]
    return out


def load_open_data_matches(
    *,
    competition_id: int,
    season_id: int,
    match_ids: list[str] | None = None,
    max_matches: int | None = None,
) -> Iterator[tuple[str, str, pd.DataFrame, pd.DataFrame, int]]:
    """Yield ``(provider, match_id, actions, frames, home_team_id)`` for the open-data competition.

    ``actions`` are SPADL (``convert_to_actions`` output, per-acting-team-LTR frame, ADR-028) with a
    ``player_name`` column attached from the events. ``frames`` is empty (event-only). ``match_ids``
    pins WHICH matches (string ids) and ``max_matches`` caps the count -- both after the manifest list.
    """
    from statsbombpy import sb  # type: ignore[import-not-found]  # optional network dep; function-local

    from scripts._sb_raw import flatten_events
    from silly_kicks.spadl import statsbomb as sb_convert

    matches = sb.matches(competition_id=competition_id, season_id=season_id, fmt="dict")
    ids = [str(k) for k in matches]
    if match_ids is not None:
        wanted = {str(m) for m in match_ids}
        ids = [i for i in ids if i in wanted]
    if max_matches is not None:
        ids = ids[:max_matches]

    for mid in ids:
        m = matches[int(mid)]
        home = int(m["home_team"]["home_team_id"])
        events = _values(sb.events(match_id=int(mid), fmt="dict"))
        id2name = _player_id_to_name(events)
        actions, _report = sb_convert.convert_to_actions(flatten_events(events, int(mid)), home_team_id=home)
        actions = actions.copy()
        pid = actions["player_id"]
        actions["player_name"] = [id2name.get(int(x)) if pd.notna(x) else None for x in pid]
        yield "statsbomb", str(mid), actions, pd.DataFrame(), home
