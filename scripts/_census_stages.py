"""ADR-129: the reusable `refs`, `fit_prepass`, `prefit` and `score` stages of
`scripts.build_territory_ranking_census`; their closure must not contain the driver (`test_stage_modules`).
The fit/report-side stages (bootstrap/report/census) stay in the driver and import `_done_results` from
here. The census bare ``from _crossed_icc import ...`` stays in the driver (report-side only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts._stages import ItemStage, Stage
from silly_kicks.id_compat import canonical_id, canonical_id_series
from silly_kicks.spadl import SPADL_COLUMNS
from silly_kicks.territory import TR_PASSES_AIMED_INTO_HULL, TR_XT_PREVENTED_ABOVE_EXPECTATION

_METRIC_COL = TR_XT_PREVENTED_ABOVE_EXPECTATION
_VOLUME_COL = TR_PASSES_AIMED_INTO_HULL
_SHARD_SCHEMA_VERSION = "territory-ranking-census-1"
_EMITTED_SHARD_COLUMNS = ["game_id", "player_id", "team_id", _METRIC_COL, _VOLUME_COL]
_FIT_SHARD_SCHEMA_VERSION = "territory-ranking-census-fit-1"

_CORPUS_PARAMS = (
    "all_competitions",
    "competitions_json",
    "competition_id",
    "season_id",
    "max_matches",
    "fit_fraction",
    "match_ids_json",
)


def _lineup_team_map(match_id: str) -> dict:
    """``{canonical player_id -> raw team_id}`` from the public StatsBomb lineups (defender identity).

    ``statsbombpy`` reads the redistributable open-data repo when no credentials are set (the
    ``assert_statsbomb_open_data_mode`` guard runs before any corpus pull). A player is on exactly one
    team per match, so the lineup fully determines each defender's team -- the identity the metric
    output (keyed on ``player_id`` only) does not carry.

    The ``sb.lineups(fmt="dict")`` payload shape VARIES across statsbombpy versions / open-data
    competitions, and this MUST reach the real player dicts in every case (a shape whose players it
    cannot reach silently maps ZERO defenders -- worse than a crash, because it looks like a match with
    no defenders rather than a parse failure):

    1. The confirmed real shape (verified on the live corpus, match 3879673):
       ``{team_id(int): {"team_id":..., "team_name":..., "lineup":[player_dict, ...]}}`` -- the players
       live in ``roster["lineup"]``; the team id is ``roster["team_id"]`` (falling back to the
       top-level key, which IS the team id int, if the roster dict lacks it).
    2. The list form (WC2022): ``{team_name: [player_dict, ...]}`` -- players are the list; team id is
       each player dict's own ``team_id``.
    3. The pid-keyed form: ``{team_name: {player_id: player_dict}}`` -- players are ``roster.values()``;
       team id is each player dict's own ``team_id``.

    A stray non-dict entry INSIDE a resolved players iterable is skipped (a defensive backstop only);
    the primary logic reaches the real player dicts, so real defenders still map.
    """
    from statsbombpy import sb  # type: ignore[import-not-found]  # optional network dep; function-local

    lineups = sb.lineups(match_id=int(match_id), fmt="dict")
    out: dict = {}
    for top_key, roster in lineups.items():
        # Resolve the iterable of PLAYER DICTS + the roster-level team id (shape 1's "lineup"/"team_id"
        # wrapper vs the bare list/pid-keyed forms of shapes 2 & 3).
        if isinstance(roster, dict) and "lineup" in roster:
            players_iter = roster["lineup"]  # shape 1: the wrapped lineup list
        elif isinstance(roster, dict):
            players_iter = roster.values()  # shape 3: pid-keyed player dicts
        else:
            players_iter = roster  # shape 2: a bare list of player dicts

        if isinstance(roster, dict) and "team_id" in roster:
            team_id_from_roster = roster["team_id"]  # shape 1: the roster carries its own team id
        elif _looks_like_team_id(top_key):
            team_id_from_roster = top_key  # shape 1 fallback: the top-level key IS the team id int
        else:
            team_id_from_roster = None  # shapes 2 & 3: the top key is a team NAME -> use per-player id

        for player in players_iter:
            if not isinstance(player, dict):
                continue  # backstop: a stray non-dict entry never crashes the corpus pass
            pid = player.get("player_id")
            tid = player.get("team_id") or team_id_from_roster
            if pid is None or tid is None:
                continue
            out[canonical_id(pid)] = tid
    return out


def _looks_like_team_id(key) -> bool:
    """A top-level lineups key is a team id (shape 1) rather than a team NAME (shapes 2/3) iff it is an
    integral value -- an ``int`` or a digit string. A team NAME is a non-numeric ``str``."""
    if isinstance(key, bool):  # bool is an int subclass; a team id is never a bool
        return False
    if isinstance(key, int):
        return True
    if isinstance(key, str):
        return key.strip().lstrip("-").isdigit()
    return False


def _score_match(item, *, xt, completion_model) -> pd.DataFrame:
    """One match -> a per-``(defender, game, team)`` counterfactual-metric shard (a tidy shard).

    Runs ``compute_territorial_dominance(method="counterfactual")`` with the INJECTED (leakage-disjoint)
    ``xt`` + ``completion_model``, then joins the public lineup to stamp each defender's ``team_id``
    (the metric output is keyed on ``player_id`` only). Returns EMPTY (columns present) when the match
    yields no scored defender -- an empty shard means "ran, produced nothing" (ADR-052), never a crash.
    """
    from silly_kicks.territory import compute_territorial_dominance

    _provider, match_id, actions, *_ = item
    samples, _report = compute_territorial_dominance(
        actions, xt=xt, method="counterfactual", completion_model=completion_model
    )
    if samples.empty:
        return pd.DataFrame(columns=_EMITTED_SHARD_COLUMNS)

    team_map = _lineup_team_map(str(match_id))
    pid_canon = canonical_id_series(samples["player_id"])
    out = pd.DataFrame(
        {
            "game_id": samples["game_id"].to_numpy(),
            "player_id": samples["player_id"].to_numpy(),
            "team_id": [team_map.get(p) for p in pid_canon],
            _METRIC_COL: pd.to_numeric(samples[_METRIC_COL], errors="coerce").to_numpy(),
            _VOLUME_COL: pd.to_numeric(samples[_VOLUME_COL], errors="coerce").to_numpy(),
        }
    )
    return out.reindex(columns=_EMITTED_SHARD_COLUMNS)


def _fit_disjoint_models(fit_actions: list[pd.DataFrame]):
    """Fit the exogenous ``xt`` + ``PassCompletionModel`` on a LEAKAGE-DISJOINT fit corpus (inline).

    Self-contained (no import from the validator / Task 12): the fit corpus is a set of matches
    HELD OUT from the scored set, so no scored match's own actions train either the value model or
    the completion model. It is ONE fit over the WHOLE pooled fit corpus -- under ``--all-competitions``
    that pools across every competition/season, giving a single consistent metric scale across the
    broad corpus (the whole point of the multi-competition census: a crossed defender+team ICC is only
    interpretable when every cell is valued on the same scale). Fails loud when the fit corpus is empty
    (a caller bug, not a silent NaN).
    """
    from silly_kicks.expected_passing import PassCompletionModel
    from silly_kicks.xthreat import ExpectedThreat

    if not fit_actions:
        raise SystemExit("no fit-corpus matches: the leakage-disjoint split left nothing to fit xt/completion on")
    # Memory: xt.fit + PassCompletionModel.fit read only SPADL-canonical columns (type_id/result_id/
    # start_x/y/end_x/y), so prune each fit-match to its SPADL_COLUMNS intersection BEFORE the concat.
    # Complete-by-construction (both fits are pure SPADL consumers) + stable (SPADL_COLUMNS is the schema
    # SSOT, not a hand-list) + drops only non-canonical extras (preserve_native/enriched/tracking joins)
    # -> lower peak memory on the ~half-corpus pool. Byte-identical fit (tests/scripts/
    # test_build_territory_ranking_census::test_fit_disjoint_models_column_prune_is_byte_identical).
    _keep = list(SPADL_COLUMNS)
    pruned = [a[[c for c in _keep if c in a.columns]] for a in fit_actions]
    pooled = pd.concat(pruned, ignore_index=True)
    xt = ExpectedThreat().fit(pooled)
    completion_model = PassCompletionModel().fit(pooled)
    return xt, completion_model


def _prune_actions_slice(actions: pd.DataFrame) -> pd.DataFrame:
    """One fit-corpus match's actions pruned to their ``SPADL_COLUMNS`` intersection -- the fit-prepass
    shard (ADR-052 Task 13). ``_fit_disjoint_models`` reads only SPADL-canonical columns, so pruning at
    shard time drops non-canonical extras (lower peak fit memory) and re-pruning in ``_fit_disjoint_models``
    is a byte-identical no-op."""
    keep = [c for c in SPADL_COLUMNS if c in actions.columns]
    return actions[keep].reset_index(drop=True)


def _read_json_param(path: str | None) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def _broad(p) -> bool:
    return bool(p["all_competitions"]) or p["competitions_json"] is not None


def _corpus_declare(p) -> dict:
    """The CONTENT of the two file-valued selectors (a path alone would not key a changed file)."""
    return {
        "competitions": _read_json_param(p["competitions_json"]),
        "match_ids": _read_json_param(p["match_ids_json"]),
    }


def _source_marker(p):
    """The broad-corpus marker (4.77.1 stale-shard rule): None on the single-competition path, so its
    historical token inputs are unchanged."""
    if not _broad(p):
        return None
    if not p["competitions_json"]:
        return "open-data-all"
    return {"open-data-narrowed": sorted(tuple(c) for c in _read_json_param(p["competitions_json"]))}


def _refs_work(key, ctx) -> dict:
    """The competitions + the sorted-by-match-id leakage-disjoint split (legacy main `:546-595`)."""
    import dataclasses

    from scripts._sb_open_data import all_open_competitions, assert_statsbomb_open_data_mode, list_open_data_refs

    p = ctx.params
    if _broad(p) and (p["competition_id"] is not None or p["season_id"] is not None):
        raise ValueError(
            "--all-competitions / --competitions-json census the whole open-data corpus and cannot be "
            "combined with the single-competition --competition-id / --season-id selectors."
        )
    assert_statsbomb_open_data_mode()  # fail-closed: never pull the private API for a public artifact
    if _broad(p):
        comps = (
            [tuple(c) for c in _read_json_param(p["competitions_json"])]
            if p["competitions_json"]
            else all_open_competitions()
        )
    else:
        comps = [
            (
                43 if p["competition_id"] is None else p["competition_id"],
                106 if p["season_id"] is None else p["season_id"],
            )
        ]
    match_ids = _read_json_param(p["match_ids_json"])
    refs = list_open_data_refs(comps, match_ids=(match_ids or {}).get("statsbomb"), max_matches=p["max_matches"])
    refs_sorted = sorted(refs, key=lambda r: str(r.match_id))
    n_fit = max(1, round(len(refs_sorted) * p["fit_fraction"])) if refs_sorted else 0
    return {
        "competitions": [list(c) for c in comps],
        "fit_refs": [dataclasses.asdict(r) for r in refs_sorted[:n_fit]],
        "score_refs": [dataclasses.asdict(r) for r in refs_sorted[n_fit:]],
    }


def _refs_of(ctx, part: str):
    from scripts._sb_open_data import OpenDataRef

    return [OpenDataRef(**d) for d in ctx.upstream("refs")[part]]


def _load(ref, ctx):
    """Resume-before-download: the executor calls this only for a pending unit (ADR-052 D14)."""
    from scripts._driver_sources import resolve_cache_dir
    from scripts._sb_open_data import assert_statsbomb_open_data_mode, load_open_data_match

    assert_statsbomb_open_data_mode()  # pins the revision in THIS worker process (decision P6)
    return load_open_data_match(ref, cache_dir=resolve_cache_dir(ctx.params["cache_dir"]))


def _fit_declare(p) -> dict:
    marker = _source_marker(p)
    return {
        "metric": "territory_ranking_census_fit",
        "schema": _FIT_SHARD_SCHEMA_VERSION,
        "fit_fraction": p["fit_fraction"],
        **({"source": marker} if marker is not None else {}),
    }


def _score_declare(p) -> dict:
    marker = _source_marker(p)
    return {
        "metric": "territory_ranking_census",
        "schema": _SHARD_SCHEMA_VERSION,
        "xt": "leakage_disjoint_fit",
        "fit_fraction": p["fit_fraction"],
        **({"source": marker} if marker is not None else {}),
    }


def _done_results(ctx, stage: str) -> list[pd.DataFrame]:
    """An ItemStage's unit results in sorted key order (the legacy sorted shard glob), done units only."""
    from scripts._stages import read_result, sorted_units, unit_states

    st = ctx.pipeline.stage(stage)
    gen = ctx.generation(stage)
    states = unit_states(ctx, st, sorted_units(ctx, st), verify=False)
    return [read_result(ctx.store, f"{gen}/{k}", names=states.names) for k, v in states.items() if v == "done"]


def _prefit_work(key, ctx) -> dict:
    """ONE leakage-disjoint xt + completion fit over the pooled fit corpus (legacy `:630-631`)."""
    xt, completion_model = _fit_disjoint_models(_done_results(ctx, "fit_prepass"))
    return {"xt": xt.to_dict(), "completion": completion_model.to_dict()}


_MODELS: dict = {}


def _models(ctx):
    """The prefit xt + completion model, rebuilt ONCE per worker (keyed by the prefit token)."""
    from silly_kicks.expected_passing import PassCompletionModel
    from silly_kicks.xthreat import ExpectedThreat

    tok = ctx.tokens["prefit"]
    if tok not in _MODELS:
        d = ctx.upstream("prefit")
        _MODELS.clear()
        _MODELS[tok] = (ExpectedThreat.from_dict(d["xt"]), PassCompletionModel.from_dict(d["completion"]))
    return _MODELS[tok]


def _score_work(item, ctx) -> pd.DataFrame:
    xt, completion_model = _models(ctx)
    return _score_match(item, xt=xt, completion_model=completion_model)


#: The reusable stages (ADR-129); the lambdas are declared here (``test_stage_modules``).
_REFS = Stage(
    name="refs",
    units=lambda ctx: ["only"],
    work=_refs_work,
    reduce=lambda r, ctx: r["only"],
    params=_CORPUS_PARAMS,
    declare=_corpus_declare,
    uses_dataset=True,
    dataset_files=lambda ctx: (),  # open data over HTTP: no dataset file
)
_FIT_PREPASS = ItemStage(
    name="fit_prepass",
    refs=lambda ctx: _refs_of(ctx, "fit_refs"),
    key=lambda ref: ref.key,
    load=_load,
    work=lambda item, ctx: _prune_actions_slice(item[2]),
    needs=("refs",),
    declare=_fit_declare,
    uses_dataset=True,
    max_failures=None,  # the legacy for_each recorded failures and fit over the rest
)
_PREFIT = Stage(
    name="prefit",
    units=lambda ctx: ["only"],
    work=_prefit_work,
    needs=("fit_prepass",),
    reduce=lambda r, ctx: r["only"],
)
_SCORE = ItemStage(
    name="score",
    refs=lambda ctx: _refs_of(ctx, "score_refs"),
    key=lambda ref: str(ref.match_id),  # the legacy key: the score shard filenames are unchanged
    load=_load,
    work=_score_work,
    needs=("refs", "prefit"),
    declare=_score_declare,
    uses_dataset=True,
    max_failures=None,  # the legacy for_each recorded failures and censused the rest
)
STAGES = {"refs": _REFS, "fit_prepass": _FIT_PREPASS, "prefit": _PREFIT, "score": _SCORE}
