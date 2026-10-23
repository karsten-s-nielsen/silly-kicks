"""ADR-129: the reusable `refs` and `extract` stages of `scripts.train_win_probability`; their closure
must not contain the trainer (`test_stage_modules`). The fit stage stays in the trainer and reads the
pooled shards + the committed `refs` output; it uses none of these helpers."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from scripts._stages import ItemStage, Stage

_DEFAULT_COMPETITION_ID = 43
_DEFAULT_SEASON_ID = 106

_SHARD_COLUMNS = [
    "game_id",
    "action_id",
    "period_id",
    "team_id",
    "player_id",
    "time_seconds",
    "type_id",
    "result_id",
    "xg",
    "home_team_id",
    "match_date",
]


def match_shard(actions: pd.DataFrame, home_team_id: int | str, match_date: str) -> pd.DataFrame:
    """One match's win-prob action rows + constants (pure). ``action_id`` is REQUIRED by
    ``compute_win_probability`` (a WIN_PROBABILITY_KEYS column) -- omitting it is what the DGX
    validation run surfaced as a KeyError the synthetic fixtures never hit."""
    cols = [
        c
        for c in (
            "game_id",
            "action_id",
            "period_id",
            "team_id",
            "player_id",
            "time_seconds",
            "type_id",
            "result_id",
            "xg",
        )
        if c in actions.columns
    ]
    out = actions[cols].copy()
    if "player_id" not in out.columns:
        out["player_id"] = np.nan
    if "xg" not in out.columns:
        out["xg"] = np.nan
    out["home_team_id"] = home_team_id
    out["match_date"] = str(match_date)
    return out.reset_index(drop=True)[_SHARD_COLUMNS]


def _competitions(params) -> list[tuple[int, int]]:
    import scripts._sb_open_data as sbod

    if params["all_competitions"] and (params["competition_id"] is not None or params["season_id"] is not None):
        raise ValueError("--all-competitions is mutually exclusive with --competition-id/--season-id")
    sbod.assert_statsbomb_open_data_mode()
    if params["all_competitions"]:
        return [(int(c[0]), int(c[1])) for c in sbod.all_open_competitions()]
    return [
        (
            _DEFAULT_COMPETITION_ID if params["competition_id"] is None else params["competition_id"],
            _DEFAULT_SEASON_ID if params["season_id"] is None else params["season_id"],
        )
    ]


def _refs_work(key, ctx) -> dict:
    """The competition set + every match reference, as one JSON unit (the legacy listing: ``:259-291`` at
    ``cd2ebcb``)."""
    import scripts._sb_open_data as sbod

    comps = _competitions(ctx.params)
    refs = sbod.list_open_data_refs(comps, max_matches=ctx.params["max_matches"])
    return {"competitions": [list(c) for c in sorted(comps)], "refs": [dataclasses.asdict(r) for r in refs]}


def _extract_refs(ctx):
    from scripts._sb_open_data import OpenDataRef

    return [OpenDataRef(**d) for d in ctx.upstream("refs")["refs"]]


def _load_match(ref, ctx):
    """Resume-before-download: the executor calls this only for a pending unit (ADR-052 D14)."""
    from scripts._driver_sources import resolve_cache_dir
    from scripts._sb_open_data import assert_statsbomb_open_data_mode, load_open_data_match

    assert_statsbomb_open_data_mode()  # pins the revision in THIS worker process (decision P6)
    return load_open_data_match(ref, cache_dir=resolve_cache_dir(ctx.params["cache_dir"])), ref.match_date


def _extract_work(item, ctx):
    lm, match_date = item
    # The legacy date_map came from the same sb.matches manifest the ref was built from.
    return match_shard(lm.actions, lm.home_team_id, match_date)


def _extract_reduce(results, ctx):
    from scripts._stages import combine_frames

    frames = list(results.values())  # sorted key order == the legacy sorted shard glob
    return combine_frames(frames, ids=("game_id",)) if frames else pd.DataFrame(columns=_SHARD_COLUMNS)


#: The reusable `refs` and `extract` stages (ADR-129); the lambdas are declared here (``test_stage_modules``).
_REFS = Stage(
    name="refs",
    units=lambda ctx: ["only"],
    work=_refs_work,
    reduce=lambda r, ctx: r["only"],
    params=("all_competitions", "competition_id", "season_id", "max_matches"),
    uses_dataset=True,
    dataset_files=lambda ctx: (),  # open data over HTTP: no dataset file
)
_EXTRACT = ItemStage(
    name="extract",
    refs=_extract_refs,
    key=lambda ref: ref.key,
    load=_load_match,
    work=_extract_work,
    needs=("refs",),
    reduce=_extract_reduce,
    declare=lambda p: {"model": "WinProbabilityModel", "cols": list(_SHARD_COLUMNS)},
    uses_dataset=True,
    max_failures=None,  # legacy for_each recorded failures and fit over the rest
)
STAGES = {"refs": _REFS, "extract": _EXTRACT}
