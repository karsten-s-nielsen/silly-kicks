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

import dataclasses
import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import pandas as pd

from scripts._loader_pining import LoadedMatch, resolve_cache_dir

#: FIFA World Cup 2022 (male) -- the spec's corpus; the locked elite-defender prior matches it.
WORLD_CUP_2022 = (43, 106)


def assert_statsbomb_open_data_mode() -> None:
    """Fail-closed public-only guard: refuse to run if StatsBomb CREDENTIALS are configured.

    ``statsbombpy`` reads from the redistributable OPEN-data repo (github.com/statsbomb/open-data)
    ONLY when no credentials are set; with ``SB_USERNAME`` / ``SB_PASSWORD`` present it pulls the
    PRIVATE API, whose matches are NOT redistributable. A public, reproducible artifact must never be
    built from that -- so this raises rather than silently stamp private data as ``public``.
    """
    if os.environ.get("SB_USERNAME") or os.environ.get("SB_PASSWORD"):
        raise SystemExit(
            "StatsBomb credentials (SB_USERNAME/SB_PASSWORD) are set -> statsbombpy would pull the "
            "PRIVATE API. This reliability artifact is public-only (open data, reproducible by "
            "anyone); refusing to run. Unset the credentials to use the open-data corpus."
        )


def all_open_competitions() -> list[tuple[int, int]]:
    """Every ``(competition_id, season_id)`` in the StatsBomb OPEN-data manifest (fail-closed public).

    In open-data mode ``sb.competitions()`` returns exactly the redistributable public releases (WC
    2018/2022, the Euros, the Women's World Cup, FA WSL, La Liga, UCL finals, NWSL, ...) -- thousands
    of matches. This is the broad default corpus for the reliability study (a single tournament is far
    too thin for a team-discrimination ICC / split-half); ``assert_statsbomb_open_data_mode`` guards
    that the manifest is the OPEN one.
    """
    from statsbombpy import sb  # type: ignore[import-not-found]  # optional network dep; function-local

    assert_statsbomb_open_data_mode()
    comps = sb.competitions(fmt="dict")
    return sorted(
        {(int(c["competition_id"]), int(c["season_id"])) for c in _values(comps)}  # type: ignore[index]
    )


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


def _event_id_to_xg(events: list[dict]) -> dict[str, float]:
    """``event_id -> shot.statsbomb_xg`` for every shot event (StatsBomb's own pre-shot xG).

    Nested (``e["shot"]["statsbomb_xg"]``), so it is NOT reachable via ``flatten_events``'s top-level
    ``surface_native``; the loader joins it onto SPADL actions by ``original_event_id`` instead.
    """
    out: dict[str, float] = {}
    for e in events:
        shot = e.get("shot")
        if isinstance(shot, dict) and shot.get("statsbomb_xg") is not None and e.get("id") is not None:
            out[str(e["id"])] = float(shot["statsbomb_xg"])
    return out


@dataclasses.dataclass(frozen=True)
class OpenDataRef:
    """A cheap reference to one StatsBomb open-data match (ADR-052 D14).

    ``home_team_id`` and ``match_date`` come from the competition manifest the listing already fetched,
    so loading a ref needs no second manifest call. A ref IS its key (``match_date`` is excluded from
    equality) and the key is ``("statsbomb", match_id)`` -- the key every open-data driver already uses.
    """

    competition_id: int
    season_id: int
    match_id: str
    home_team_id: int
    match_date: str = dataclasses.field(default="", compare=False)

    @property
    def key(self) -> tuple[str, str]:
        return ("statsbomb", self.match_id)


def list_open_data_refs(
    competitions, *, match_ids: list[str] | None = None, max_matches: int | None = None
) -> list[OpenDataRef]:
    """Every requested open-data match, as references: the ONE multi-competition walk (spec §4.2).

    ``competitions`` is an ordered iterable of ``(competition_id, season_id)``; each manifest is listed
    once, in order, keeping manifest order within it. ``match_ids`` (string ids) filters across all
    competitions; ``max_matches`` is a GLOBAL cap -- the walk stops, without listing further
    competitions, once it is reached. Fail-closed public-only first.
    """
    from statsbombpy import sb  # type: ignore[import-not-found]  # optional network dep; function-local

    assert_statsbomb_open_data_mode()
    wanted = None if match_ids is None else {str(m) for m in match_ids}
    refs: list[OpenDataRef] = []
    for competition_id, season_id in competitions:
        manifest = sb.matches(competition_id=int(competition_id), season_id=int(season_id), fmt="dict")
        for raw_id, m in manifest.items():
            mid = str(raw_id)
            if wanted is not None and mid not in wanted:
                continue
            if max_matches is not None and len(refs) >= max_matches:
                return refs
            refs.append(
                OpenDataRef(
                    int(competition_id),
                    int(season_id),
                    mid,
                    int(m["home_team"]["home_team_id"]),
                    str(m.get("match_date", "")),
                )
            )
    return refs


def _cached_json(path: Path | None, fetch) -> list[dict]:
    """``fetch()`` through a JSON cache file: read it when present, else fetch and write it atomically.

    With no path this fetches every time (today's behaviour). The first fetch is written atomically
    (temp + ``os.replace``) so a killed run never leaves a torn cache entry; the JSON round-trip of
    a JSON-parsed StatsBomb payload is the identity, so a cached load builds what a fetched one does.
    """
    if path is not None and path.is_file() and path.stat().st_size > 0:
        return json.loads(path.read_text(encoding="utf-8"))
    payload = fetch()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.partial")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)
    return payload


def _open_events(match_id: str, cache_root: Path | None) -> list[dict]:
    """The raw events of one match -- from ``<root>/statsbomb_open/<match_id>.json`` when cached."""
    from statsbombpy import sb  # type: ignore[import-not-found]

    path = None if cache_root is None else Path(cache_root) / "statsbomb_open" / f"{match_id}.json"
    return _cached_json(path, lambda: _values(sb.events(match_id=int(match_id), fmt="dict")))


def _open_frames(match_id: str, cache_root: Path | None) -> list[dict]:
    """The raw 360 freeze-frames of one match -- ``<root>/statsbomb_open/<match_id>.360.json`` when
    cached (a distinct filename from the events, so the two never collide in one cache dir)."""
    from statsbombpy import sb  # type: ignore[import-not-found]

    path = None if cache_root is None else Path(cache_root) / "statsbomb_open" / f"{match_id}.360.json"
    return _cached_json(path, lambda: _values(sb.frames(match_id=int(match_id), fmt="dict")))


def fetch_open_360_raw(match_id, *, cache_dir=None) -> tuple[list[dict], list[dict]]:
    """Raw ``(events, freeze_frames)`` of one open-data match, each JSON-cached under ``cache_dir``
    (else ``$SILLY_KICKS_CORPUS_CACHE_DIR``, else fetched every time). The raw-360 coverage driver
    (``scripts/build_sb360_coverage.py``) consumes both lists; fail-closed public-only before any fetch.
    """
    assert_statsbomb_open_data_mode()
    root = resolve_cache_dir(cache_dir)
    return _open_events(str(match_id), root), _open_frames(str(match_id), root)


def load_open_data_match(ref: OpenDataRef, *, preserve_native: tuple[str, ...] = (), cache_dir=None) -> LoadedMatch:
    """Load ONE open-data match: SPADL actions + ``player_name`` + StatsBomb's own shot ``xg``.

    Inherently event-only: ``frames`` is the empty DataFrame it always was. The raw events are cached
    under ``cache_dir`` (else ``$SILLY_KICKS_CORPUS_CACHE_DIR``, else not at all) so resumes and repeat
    battery runs stop re-fetching from GitHub. Fail-closed public-only before any fetch.
    """
    from scripts._sb_raw import flatten_events
    from silly_kicks.spadl import statsbomb as sb_convert

    assert_statsbomb_open_data_mode()
    events = _open_events(ref.match_id, resolve_cache_dir(cache_dir))
    id2name = _player_id_to_name(events)
    flat = flatten_events(events, int(ref.match_id), surface_native=preserve_native)
    actions, _report = sb_convert.convert_to_actions(
        flat, home_team_id=ref.home_team_id, preserve_native=list(preserve_native) or None
    )
    actions = actions.copy()
    pid = actions["player_id"]
    actions["player_name"] = [id2name.get(int(x)) if pd.notna(x) else None for x in pid]
    # StatsBomb's own pre-shot xG, joined onto shot rows by original_event_id (NaN elsewhere) so a
    # consumer can pass xg_column="xg" -- e.g. the reliability study's high_opportunity_shots KPI.
    id2xg = _event_id_to_xg(events)
    actions["xg"] = actions["original_event_id"].astype(str).map(id2xg).astype("float64")
    return LoadedMatch("statsbomb", str(ref.match_id), actions, pd.DataFrame(), ref.home_team_id, None, None)


def open_data_source(
    competitions,
    *,
    match_ids: list[str] | None = None,
    max_matches: int | None = None,
    preserve_native: tuple[str, ...] = (),
    cache_dir=None,
) -> tuple[list[OpenDataRef], Callable[[OpenDataRef], LoadedMatch]]:
    """The open-data source factory: ``(refs, load)`` for ``for_each`` (ADR-052 D14, owner-ratified
    reuse 2026-09-23).

    ``refs`` is exactly ``list_open_data_refs(competitions, …)`` (the ONE multi-competition walk);
    ``load(ref)`` is exactly ``load_open_data_match(ref, …)`` with the cache dir resolved ONCE here.
    Gate-clean: it lists refs and loads one match, so it is neither a stream loader (Rule A) nor a
    loading loop (Rule C). Symmetric with ``scripts._loader_pining.pining_source``.
    """
    cd = resolve_cache_dir(cache_dir)
    refs = list_open_data_refs(competitions, match_ids=match_ids, max_matches=max_matches)

    def load(ref: OpenDataRef) -> LoadedMatch:
        return load_open_data_match(ref, preserve_native=preserve_native, cache_dir=cd)

    return refs, load


def load_open_data_matches(
    *,
    competition_id: int,
    season_id: int,
    match_ids: list[str] | None = None,
    max_matches: int | None = None,
    preserve_native: tuple[str, ...] = (),
) -> Iterator[tuple[str, str, pd.DataFrame, pd.DataFrame, int]]:
    """Yield ``(provider, match_id, actions, frames, home_team_id)`` for one open-data competition.

    A byte-identical wrapper over ``list_open_data_refs`` + ``load_open_data_match`` (ADR-052 D14).
    ``actions`` are SPADL with a ``player_name`` column and an ``xg`` column (StatsBomb's own pre-shot
    xG on shots, NaN elsewhere); ``frames`` is empty (event-only). Corpus drivers must not call it (CI
    Rule A); they list refs and load one match per ``for_each`` item.
    """
    for ref in list_open_data_refs([(competition_id, season_id)], match_ids=match_ids, max_matches=max_matches):
        m = load_open_data_match(ref, preserve_native=preserve_native)
        yield m.provider, m.match_id, m.actions, cast("pd.DataFrame", m.frames), ref.home_team_id
