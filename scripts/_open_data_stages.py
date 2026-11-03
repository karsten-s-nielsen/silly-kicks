"""The open-data `refs` and `extract` stage parts shared by the public event-only pipelines (win_probability,
pass_completion, xsuccess; ADR-128). The parameters mirror the legacy CLIs: ``--all-competitions`` or one
``--competition-id``/``--season-id`` release (default WC2022), ``--max-matches`` and ``--cache-dir``."""

from __future__ import annotations

import dataclasses

from scripts._stages import Param

#: WC2022 (competition 43, season 106): the single-release default.
DEFAULT_COMPETITION = (43, 106)

PARAMS = (
    Param("dataset", str, "sb_open", help="declared dataset name (scripts/hf_jobs/datasets.toml)"),
    Param("all_competitions", flag=True, help="full public open-data corpus (every release)"),
    Param("competition_id", int, None, help="open-data competition (default 43 = World Cup)"),
    Param("season_id", int, None, help="open-data season (default 106 = 2022)"),
    Param("max_matches", int, None, help="cap the number of matches"),
    Param("cache_dir", str, None, help="raw open-data events cache root (else $SILLY_KICKS_CORPUS_CACHE_DIR)"),
)

#: The `refs` stage's declared parameters.
REFS_PARAMS = ("all_competitions", "competition_id", "season_id", "max_matches")


def competitions(params) -> list[tuple[int, int]]:
    """The releases to walk: every open-data release, or the one named (default WC2022)."""
    import scripts._sb_open_data as sbod

    if params["all_competitions"] and (params["competition_id"] is not None or params["season_id"] is not None):
        raise ValueError("--all-competitions is mutually exclusive with --competition-id/--season-id")
    sbod.assert_statsbomb_open_data_mode()
    if params["all_competitions"]:
        return [(int(c[0]), int(c[1])) for c in sbod.all_open_competitions()]
    cid, sid = DEFAULT_COMPETITION
    return [(cid if params["competition_id"] is None else params["competition_id"],
             sid if params["season_id"] is None else params["season_id"])]  # fmt: skip


def refs_work(key, ctx) -> dict:
    """The competition set and every match reference, as one JSON unit."""
    import scripts._sb_open_data as sbod

    comps = competitions(ctx.params)
    refs = sbod.list_open_data_refs(comps, max_matches=ctx.params["max_matches"])
    return {"competitions": [list(c) for c in sorted(comps)], "refs": [dataclasses.asdict(r) for r in refs]}


def extract_refs(ctx):
    from scripts._sb_open_data import OpenDataRef

    return [OpenDataRef(**d) for d in ctx.upstream("refs")["refs"]]


def load_open_match(ref, ctx):
    """Resume-before-download: the executor calls this only for a pending unit (ADR-052 D14)."""
    from scripts._driver_sources import resolve_cache_dir
    from scripts._sb_open_data import assert_statsbomb_open_data_mode, load_open_data_match

    assert_statsbomb_open_data_mode()  # pins the revision in THIS worker process (decision P6)
    return load_open_data_match(ref, cache_dir=resolve_cache_dir(ctx.params["cache_dir"]))


def corpus_label(params, comps) -> str:
    """The legacy corpus label: every release (``--all-competitions``), or the one release."""
    if params["all_competitions"]:
        return f"statsbomb-open (all {len(comps)} open-data competitions)"
    return f"statsbomb-open (competition {comps[0][0]}, season {comps[0][1]})"


def pass_fields(ctx) -> dict:
    """The extract pass in ``metrics.json`` (decision P4, ADR-128): corpus-scoped counters, never pass-scoped
    ones, so resume history cannot change the file (INV-1); ``generation`` is the stage token."""
    summary = ctx.summary("extract")
    return {
        "generation": ctx.tokens["extract"],
        "n_attempted": summary["n_units"],
        "n_failed": summary["n_failed"],
        "n_counters_unrecorded": 0,
        "n_excluded": summary["n_excluded"],
    }
