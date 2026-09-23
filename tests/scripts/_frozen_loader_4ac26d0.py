"""Frozen VERBATIM copies of the ``4ac26d0`` ``load_matches`` / ``load_statsbomb_matches`` bodies.

The live wrappers are refactored over ``list_match_refs`` + ``load_match``; these frozen references
prove the refactor changed no OBSERVABLE output (tuples, the ``EXCLUDED ...`` stderr line, the
``excluded n/m`` summary) over a monkeypatched network (spec §4.1, CDLS-SPEC-29). They reference the
live module's helpers by attribute (``lp.X``) so a test's monkeypatch of those helpers reaches both
the live wrapper and this reference.

tests/scripts/ has NO __init__.py; conftest puts scripts/ on sys.path.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator

import _loader_pining as lp
import _sb_open_data as sbod
import pandas as pd


def load_matches_4ac26d0(
    *,
    providers: list[str],
    match_ids: dict[str, list[str]] | None = None,
    token: str | None = None,
    tracking_limit: int | None = None,
    max_per_provider: int | None = None,
    cache_dir=None,
) -> Iterator[tuple]:
    tok, base_url = lp._resolve_token(token), lp._base_url()
    n_total = 0
    n_excluded = 0
    for provider in providers:
        manifest = {m["id"]: m for m in lp._list_matches(provider, tok, base_url)}
        wanted = lp._wanted_for_provider(list(manifest), provider, match_ids, max_per_provider)
        for match_id in wanted:
            n_total += 1
            artifacts = manifest[match_id]["artifacts"]
            actions, frames, home, _visible_area, report = lp._build_match_with_retry(
                provider, match_id, artifacts, tok, base_url, tracking_limit, cache_dir=cache_dir
            )
            if provider == "skillcorner" and getattr(report, "geometry_excluded", False):
                reason = getattr(report, "geometry_reason", "")
                print(f"  EXCLUDED {provider}/{match_id}: {reason}", file=sys.stderr)
                n_excluded += 1
                continue
            yield provider, match_id, actions, frames, home
    print(f"excluded {n_excluded}/{n_total} matches", file=sys.stderr)


def load_statsbomb_matches_4ac26d0(
    *,
    match_ids: list[str] | None = None,
    token: str | None = None,
    max_matches: int | None = None,
    cache_dir=None,
) -> Iterator[tuple]:
    tok, base_url = lp._resolve_token(token), lp._base_url()
    manifest = {m["id"]: m for m in lp._list_matches("statsbomb", tok, base_url)}
    wanted = lp._wanted_for_provider(
        list(manifest), "statsbomb", {"statsbomb": match_ids} if match_ids else None, max_matches
    )
    for match_id in wanted:
        artifacts = manifest[match_id]["artifacts"]
        actions, frames, home, visible_area, _report = lp._build_match_with_retry(
            "statsbomb", match_id, artifacts, tok, base_url, None, cache_dir=cache_dir
        )
        yield "statsbomb", match_id, actions, frames, home, visible_area


def load_open_data_matches_4ac26d0(
    *, competition_id: int, season_id: int, match_ids=None, max_matches=None, preserve_native=()
) -> Iterator[tuple]:
    from statsbombpy import sb  # type: ignore[import-not-found]

    from scripts._sb_raw import flatten_events
    from silly_kicks.spadl import statsbomb as sb_convert

    sbod.assert_statsbomb_open_data_mode()
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
        events = sbod._values(sb.events(match_id=int(mid), fmt="dict"))
        id2name = sbod._player_id_to_name(events)
        flat = flatten_events(events, int(mid), surface_native=preserve_native)
        actions, _report = sb_convert.convert_to_actions(
            flat, home_team_id=home, preserve_native=list(preserve_native) or None
        )
        actions = actions.copy()
        pid = actions["player_id"]
        actions["player_name"] = [id2name.get(int(x)) if pd.notna(x) else None for x in pid]
        id2xg = sbod._event_id_to_xg(events)
        actions["xg"] = actions["original_event_id"].astype(str).map(id2xg).astype("float64")
        yield "statsbomb", str(mid), actions, pd.DataFrame(), home
