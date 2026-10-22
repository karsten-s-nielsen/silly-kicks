"""TF-54b defender-RANKING census + gate + ranking (owner-run, ADR-099 ranking gate).

Decides -- on MEASURED evidence -- whether a per-defender RANKING is licensed on a corpus for the
``territory`` counterfactual metric (``territory_xt_prevented_above_expectation``). A per-defender
territorial number is TEAM-CONDITIONED by construction (see ``scripts/_crossed_icc.py``), so a
ranking is only defensible when the corpus can (a) SEPARATE a defender from the team they play for --
which needs defenders observed on >= 2 distinct teams -- and (b) show a defender-share ICC whose
bootstrap lower bound clears a floor with adequate power. Both are LOCKED, universal-safe thresholds
committed BEFORE the owner run (the TF-19 ``NAMED_KEEPER_PRIOR`` idiom), and ``census_gate`` reads
ONLY those constants -- never an inline literal.

The pure core (``tier1_counts`` / ``tier2_icc`` / ``census_gate`` / ``build_ranking``) is unit-tested
with NO network in ``tests/scripts/test_build_territory_ranking_census.py``; the stage pipeline is the
I/O (ADR-128): clean-tree guard FIRST (ADR-037, the stage CLI), fail-closed public-only StatsBomb open
data (``assert_statsbomb_open_data_mode``), per-match counterfactual compute as ``ItemStage`` units
(ADR-052) with an ``xt`` + ``PassCompletionModel`` fit on a LEAKAGE-DISJOINT fit split (self-contained --
no forward dependency on the validator), the Tier-2 bootstrap replicates on the InnerPool from ONE seeded
stream, and an input contract (ADR-056). ``statsbombpy`` is an optional ``scripts/`` network dep for the
public lineups (defender -> team identity).

HONEST LIMIT (recorded in the artifact): a NEGATIVE verdict is the expected outcome on a single-club /
national-team corpus, where one player = one team and the defender-vs-team confound is unidentifiable.
The census exists to make that verdict MEASURED rather than assumed -- and to license the ranking on
the day a multi-club transfer corpus supplies the cross-team observations it needs.

The census aggregates exactly THIS run's units (decision P4): the corpus selectors choose the
``refs`` stage's match list, and every downstream stage reads only those units -- a narrowed run and a
full run never combine, even in the same store (the ADR-052 reconcile-superset behavior is gone with the
``--out``-keyed shard directory).

Usage (owner; ADR-128 stage contract, the uniform CLI from ``scripts._stages.main``)::

    python -m scripts.build_territory_ranking_census --all-competitions \
        --store <DIR> --run-all --out docs/research/territory_ranking_census
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `from scripts._X` resolves
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/, so the bare `_crossed_icc` import resolves

from _crossed_icc import bootstrap_icc_and_power  # scripts/ on sys.path (conftest / the line above)

from scripts._input_contract import declare_inputs
from scripts._stages import FileSet, ItemStage, Param, Pipeline, Stage, main
from silly_kicks.id_compat import canonical_id, canonical_id_series
from silly_kicks.spadl import SPADL_COLUMNS
from silly_kicks.territory import (
    TR_PASSES_AIMED_INTO_HULL,
    TR_XT_PREVENTED_ABOVE_EXPECTATION,
)

# --------------------------------------------------------------------------------------------------
# LOCKED census constants (committed BEFORE the owner run; the gate reads ONLY these -- never a
# literal in the gate body). Universal-safe: chosen so a false NEGATIVE (refuse a ranking that was
# actually licensable) is the safe direction -- a wrong ranking is worse than a withheld one.
# --------------------------------------------------------------------------------------------------

#: The >=2-team identifiability precondition. A defender-vs-team decomposition needs defenders
#: observed on >= 2 DISTINCT teams; below this many such defenders the crossed model cannot separate
#: the two sources at all (the Task-10 ICC is unidentifiable). 20 is a conservative floor -- enough
#: crossed observations for the Henderson-III moment estimator to have measurable between-source df,
#: well above the handful that would make the ICC a single-cell artifact.
MIN_MULTI_TEAM_DEFENDERS: int = 20

#: A (defender, team) cell counts toward Tier-1 coverage only when the defender faced at least this
#: many hull-aimed opponent passes on that team -- the counterfactual scoring denominator
#: (``territory_passes_aimed_into_hull``). 30 is the standard "enough events to estimate a rate"
#: floor used across this repo's reliability work; a thinner cell is a noise estimate, not a signal.
MIN_PASSES_FACED: int = 30

#: The defender-share ICC bootstrap LOWER bound must exceed this for the defender signal to be
#: distinguishable from zero. 0.0 is the honest floor: "the interval clears zero" is exactly the
#: identifiability claim (a defender-null design has lo <= 0). A positive floor would demand a
#: pre-judged effect magnitude the corpus has not yet supplied; 0.0 tests detectability, not size.
ICC_LOWER_FLOOR: float = 0.0

#: Required bootstrap power at ``ICC_EFFECT_SIZE``: the fraction of bootstrap ICC draws that clear the
#: effect size must reach this. 0.8 is the conventional power floor -- an under-powered design cannot
#: be trusted to detect a real defender share even where one exists.
POWER_FLOOR: float = 0.8

#: The defender-share ICC effect size the design must be powered to DETECT. 0.05 is a deliberately
#: small "any meaningful defender signal" threshold (5% of the between-unit variance attributable to
#: the defender) -- large enough to matter, small enough that a design failing to detect it is
#: genuinely under-powered rather than merely facing a tiny true effect.
ICC_EFFECT_SIZE: float = 0.05

#: Bootstrap settings for the Tier-2 ICC interval + power proxy (deterministic).
_N_BOOT: int = 400
_ALPHA: float = 0.10  # a two-sided 90% percentile interval (lo = 5th percentile)
_BOOT_SEED: int = 0

_METRIC_COL = TR_XT_PREVENTED_ABOVE_EXPECTATION
_VOLUME_COL = TR_PASSES_AIMED_INTO_HULL


# --------------------------------------------------------------------------------------------------
# Pure core (CI-tested, NO network).
# --------------------------------------------------------------------------------------------------
def tier1_counts(defender_team_metric: pd.DataFrame, *, min_passes_faced: int) -> dict:
    """Tier-1 identifiability + coverage counts from the per-``(defender, game, team)`` metric table.

    Parameters
    ----------
    defender_team_metric : pd.DataFrame
        One row per ``(defender, game, team)`` carrying ``player_id`` / ``team_id`` and the volume
        column ``territory_passes_aimed_into_hull`` (the counterfactual scoring denominator).
    min_passes_faced : int
        A ``(defender, team)`` cell qualifies only when its summed passes-faced clears this.

    Returns
    -------
    dict
        ``n_defenders`` -- distinct defenders. ``n_multi_team_defenders`` -- defenders appearing on
        >= 2 DISTINCT teams (the crossed-model identifiability precondition). ``n_cells`` -- distinct
        ``(defender, team)`` cells. ``n_qualifying_cells`` -- cells whose summed passes-faced clears
        ``min_passes_faced``. ``n_rows`` -- input rows (conservation companion).

    Notes
    -----
    Ids are canonicalized (ADR-019) before distinctness so a mixed-dtype id is never split into two
    defenders. Conserves: ``n_multi_team_defenders <= n_defenders`` and
    ``n_qualifying_cells <= n_cells`` hold by construction.
    """
    df = defender_team_metric
    if df.empty:
        return {
            "n_defenders": 0,
            "n_multi_team_defenders": 0,
            "n_cells": 0,
            "n_qualifying_cells": 0,
            "n_rows": 0,
        }

    player = canonical_id_series(df["player_id"])
    team = canonical_id_series(df["team_id"])
    volume = pd.to_numeric(df[_VOLUME_COL], errors="coerce").fillna(0.0)

    work = pd.DataFrame({"_p": player, "_t": team, "_v": volume})
    # Drop NA-keyed rows: a defender/team we cannot identify contributes to no cell.
    work = work[work["_p"].notna() & work["_t"].notna()]

    n_defenders = int(work["_p"].nunique())
    # Distinct teams per defender -> how many defenders span >= 2.
    teams_per_def = work.groupby("_p")["_t"].nunique()
    n_multi_team = int((teams_per_def >= 2).sum())

    cell_volume = work.groupby(["_p", "_t"])["_v"].sum()
    n_cells = len(cell_volume)
    n_qualifying = int((cell_volume >= min_passes_faced).sum())

    return {
        "n_defenders": n_defenders,
        "n_multi_team_defenders": n_multi_team,
        "n_cells": n_cells,
        "n_qualifying_cells": n_qualifying,
        "n_rows": len(df),
    }


def tier2_icc(defender_team_metric: pd.DataFrame, *, replicate_map=None) -> dict:
    """Tier-2 crossed defender+team ICC on the metric value, with a bootstrap CI + power proxy.

    Runs the Task-10 ``bootstrap_icc_and_power`` (one merged CI+power bootstrap) on the metric value keyed
    by ``(defender, team)`` (ADR-019 canonical codes). Only run when Tier-1 clears -- the census
    driver gates the call.

    Returns
    -------
    dict
        ``icc`` -- the defender-share ICC point estimate. ``lo`` -- the bootstrap lower bound.
        ``power`` -- the fraction of bootstrap ICC draws that clear ``ICC_EFFECT_SIZE`` (the design's
        power to detect that effect). ``n_boot_effective`` -- finite bootstrap replicates.

    Notes
    -----
    The power proxy reuses the SAME with-replacement defender cluster bootstrap the CI is built from
    (one draw set), so ``power`` and ``lo`` are consistent by construction. An empty / single-level
    design yields ``icc``/``lo``/``power`` = NaN rather than a fabricated value.
    """
    df = defender_team_metric
    y = pd.to_numeric(df[_METRIC_COL], errors="coerce").to_numpy(dtype=float)
    defender_codes = canonical_id_series(df["player_id"]).to_numpy()
    team_codes = canonical_id_series(df["team_id"]).to_numpy()

    finite = np.isfinite(y)
    y, defender_codes, team_codes = y[finite], defender_codes[finite], team_codes[finite]
    if len(y) == 0:
        return {"icc": float("nan"), "lo": float("nan"), "power": float("nan"), "n_boot_effective": 0}

    # ONE bootstrap pass yields BOTH the CI and the power (the two former loops drew the same resamples
    # from the same seed; merging halves the crossed-model fits, byte-identical).
    res = bootstrap_icc_and_power(
        y,
        defender_codes,
        team_codes,
        n_boot=_N_BOOT,
        alpha=_ALPHA,
        effect_size=ICC_EFFECT_SIZE,
        rng_seed=_BOOT_SEED,
        replicate_map=replicate_map,  # ADR-128: replicates on the pool, combined in replicate order (E14)
    )
    return {
        "icc": res["icc"],
        "lo": res["lo"],
        "power": res["power"],
        "n_boot_effective": res["n_boot_effective"],
    }


def census_gate(census: dict) -> dict:
    """Decide whether a per-defender ranking is licensed. Reads ONLY the LOCKED module constants.

    Ranking is licensed IFF, in order:

    1. Tier-1 clears ``MIN_MULTI_TEAM_DEFENDERS`` (the >=2-team identifiability precondition), AND
    2. Tier-2's bootstrap lower bound ``lo > ICC_LOWER_FLOOR`` (a defender share distinguishable from
       zero), AND
    3. the design is powered: ``power >= POWER_FLOOR`` at ``ICC_EFFECT_SIZE``.

    Otherwise the ranking is NOT licensed and ``reason`` names the FIRST failing gate.

    Parameters
    ----------
    census : dict
        ``{"tier1": <tier1_counts dict>, "tier2": <tier2_icc dict | None>}``. ``tier2`` is ``None``
        when Tier-1 failed (the driver never runs Tier-2 then), which the gate treats as "not run".

    Returns
    -------
    dict
        ``{"ranking_licensed": bool, "reason": str}``.
    """
    t1 = census.get("tier1") or {}
    n_multi = int(t1.get("n_multi_team_defenders", 0))
    if n_multi < MIN_MULTI_TEAM_DEFENDERS:
        return {
            "ranking_licensed": False,
            "reason": (
                f"identifiability precondition unmet: {n_multi} defender(s) appear on >=2 teams, "
                f"below MIN_MULTI_TEAM_DEFENDERS={MIN_MULTI_TEAM_DEFENDERS}. The defender-vs-team "
                f"confound is unidentifiable without cross-team observations (a multi-club transfer "
                f"corpus is required)."
            ),
        }

    t2 = census.get("tier2")
    if not t2:
        return {
            "ranking_licensed": False,
            "reason": "Tier-1 cleared but Tier-2 ICC was not computed (no tier2 result present).",
        }

    lo = float(t2.get("lo", float("nan")))
    if not np.isfinite(lo) or lo <= ICC_LOWER_FLOOR:
        return {
            "ranking_licensed": False,
            "reason": (
                f"defender-share ICC lower bound lo={lo:.4f} does not exceed "
                f"ICC_LOWER_FLOOR={ICC_LOWER_FLOOR}: the defender signal is not distinguishable "
                f"from zero (team-conditioned, not defender-attributable)."
            ),
        }

    power = float(t2.get("power", float("nan")))
    if not np.isfinite(power) or power < POWER_FLOOR:
        return {
            "ranking_licensed": False,
            "reason": (
                f"design under-powered: power={power:.3f} at ICC_EFFECT_SIZE={ICC_EFFECT_SIZE} is "
                f"below POWER_FLOOR={POWER_FLOOR}."
            ),
        }

    return {
        "ranking_licensed": True,
        "reason": (
            f"licensed: {n_multi} multi-team defenders (>= {MIN_MULTI_TEAM_DEFENDERS}), ICC lo={lo:.4f} "
            f"> {ICC_LOWER_FLOOR}, power={power:.3f} >= {POWER_FLOOR}."
        ),
    }


def build_ranking(defender_team_metric: pd.DataFrame, *, licensed: bool, min_passes_faced: int) -> pd.DataFrame | None:
    """Top defenders by ``territory_xt_prevented_above_expectation`` -- ONLY when licensed.

    Returns ``None`` when the ranking is NOT licensed (a census that does not license a ranking must
    not produce one). When licensed, aggregates the metric per defender over cells that clear
    ``min_passes_faced`` in adequate pass-faced volume and returns the defenders sorted DESCENDING by
    the summed headline metric (best defender first).

    Notes
    -----
    A defender whose total hull-aimed volume is below ``min_passes_faced`` is EXCLUDED -- a
    low-volume high value is a noise estimate, not a ranking-worthy signal.
    """
    if not licensed:
        return None

    df = defender_team_metric
    player = canonical_id_series(df["player_id"])
    volume = pd.to_numeric(df[_VOLUME_COL], errors="coerce").fillna(0.0)
    metric = pd.to_numeric(df[_METRIC_COL], errors="coerce")

    work = pd.DataFrame(
        {
            "player_id": df["player_id"].to_numpy(),
            "_p": player.to_numpy(),
            "_v": volume.to_numpy(),
            "_m": metric.to_numpy(),
        }
    )
    work = work[work["_p"].notna()]

    agg = work.groupby("_p", sort=False).agg(
        player_id=("player_id", "first"),
        **{
            _VOLUME_COL: ("_v", "sum"),
            _METRIC_COL: ("_m", "sum"),
        },
    )
    adequate = agg[agg[_VOLUME_COL] >= min_passes_faced].copy()
    return adequate.sort_values(_METRIC_COL, ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------------------
# Input contract (ADR-056).
# --------------------------------------------------------------------------------------------------
def input_contract() -> dict:
    """Declare WHICH SYMBOLS these census numbers depend on (ADR-056)."""
    from dataclasses import asdict

    from silly_kicks.territory import CounterfactualParams, TerritoryParams

    return declare_inputs(
        driver="build_territory_ranking_census",
        params={
            "territory": asdict(TerritoryParams()),
            "counterfactual": asdict(CounterfactualParams()),
            "min_multi_team_defenders": MIN_MULTI_TEAM_DEFENDERS,
            "min_passes_faced": MIN_PASSES_FACED,
            "icc_lower_floor": ICC_LOWER_FLOOR,
            "power_floor": POWER_FLOOR,
            "icc_effect_size": ICC_EFFECT_SIZE,
            "n_boot": _N_BOOT,
            "alpha": _ALPHA,
            "boot_seed": _BOOT_SEED,
        },
        extractors=["silly_kicks.territory._counterfactual", "silly_kicks.territory._compute"],
        models=["scripts._crossed_icc"],
    )


# --------------------------------------------------------------------------------------------------
# Corpus extraction (owner-run; NOT CI-exercised beyond the pure core).
# --------------------------------------------------------------------------------------------------

#: The per-(defender, game, team) shard columns + the shard-generation schema token; MUST move
#: together (ADR-052 / the 4.77.1 stale-shard rule) -- the for_each fingerprint digests token_inputs
#: only, so renaming a column while leaving the token reuses stale shards.
_SHARD_SCHEMA_VERSION = "territory-ranking-census-1"
_EMITTED_SHARD_COLUMNS = ["game_id", "player_id", "team_id", _METRIC_COL, _VOLUME_COL]
#: The fit-prepass shard schema + generation token (its own generation, disjoint from the score shards).
_FIT_SHARD_SCHEMA_VERSION = "territory-ranking-census-fit-1"


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


# ---- ADR-128 stage contract --------------------------------------------------------------------------
# A legacy line reference (`:N-M`) points at this file at `cd2ebcb`, the last version before the contract.
#
# refs (one unit: the corpus + the leakage-disjoint split) -> fit_prepass[match] -> prefit (one unit: the
# disjoint xt + completion fit) -> score[match] -> bootstrap (one inner_parallel unit: Tier-1/Tier-2, the
# replicates on the InnerPool from ONE seeded stream) -> report (one unit: census.json [+ ranking.parquet]).

_PARAMS = (
    Param("dataset", str, "sb_open", help="declared dataset name (scripts/hf_jobs/datasets.toml)"),
    Param(
        "all_competitions",
        flag=True,
        help="census the FULL public open-data corpus (all_open_competitions()). REQUIRED for a crossed "
        "defender+team ICC ranking -- defenders on >=2 teams only exist ACROSS competitions/seasons, so a "
        "single-competition census can only ever return 'not licensed'. Mutually exclusive with "
        "--competition-id/--season-id.",
    ),
    Param(
        "competitions_json",
        str,
        None,
        help="path of a JSON [[competition_id, season_id], ...] narrowing the broad corpus. Mutually exclusive "
        "with --competition-id/--season-id.",
    ),
    Param("competition_id", int, None, help="open-data competition (default 43 = World Cup)"),
    Param("season_id", int, None, help="open-data season (default 106 = 2022)"),
    Param("max_matches", int, None),
    Param(
        "fit_fraction",
        float,
        0.5,
        help="fraction of matches HELD OUT to fit xt/completion (leakage-disjoint); the rest are scored.",
    ),
    Param("match_ids_json", str, None, help='path of a JSON {"statsbomb": ["3857276", ...]} pinning WHICH matches.'),
    Param("cache_dir", str, None, help="raw open-data events cache root (else $SILLY_KICKS_CORPUS_CACHE_DIR)"),
)


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
    return [read_result(ctx.store, f"{gen}/{k}") for k, v in states.items() if v == "done"]


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


#: One entry per pool child (a child serves one bootstrap): the arrays every replicate reads.
_BOOT_ARRAYS: dict = {}


def _bootstrap_chunk(arrays_path: str, picks: list) -> list:
    """A chunk of bootstrap replicates in a pool child: the arrays load ONCE per child from the unit-local
    files (never pickled per replicate), and each replicate is `_crossed_icc._replicate_icc` exactly."""
    from _crossed_icc import _replicate_icc

    if arrays_path not in _BOOT_ARRAYS:
        base = Path(arrays_path)
        y = np.load(base / "y.npy")
        codes = json.loads((base / "codes.json").read_text(encoding="utf-8"))
        d = np.array(codes["defender"], dtype=object)
        t = np.array(codes["team"], dtype=object)
        members = [np.flatnonzero(d == u) for u in np.unique(d)]  # the bootstrap's own construction
        _BOOT_ARRAYS.clear()
        _BOOT_ARRAYS[arrays_path] = (y, d, t, members)
    y, d, t, members = _BOOT_ARRAYS[arrays_path]
    return [_replicate_icc(y, d, t, members, np.asarray(picked)) for picked in picks]


#: Replicates per pool task (400 replicates -> 8 tasks: the stage's inner width).
_BOOT_CHUNK = 50


def _replicate_map(pool, work: Path):
    """A `bootstrap_icc_and_power` replicate_map over ``pool``: chunks of `_BOOT_CHUNK` replicates, the arrays
    persisted ONCE to ``work``; results returned per replicate key (combined by the caller IN ORDER)."""
    from scripts._stages import int_key

    def rmap(task):
        _fn, items = task
        keys = sorted(items)
        if not keys:
            return {}
        y, d, t, _members, _picked = items[keys[0]]
        work.mkdir(parents=True, exist_ok=True)
        np.save(work / "y.npy", np.asarray(y, dtype=float))
        (work / "codes.json").write_text(json.dumps({"defender": d.tolist(), "team": t.tolist()}), encoding="utf-8")
        chunks = {
            int_key(c): (str(work), [items[k][4].tolist() for k in keys[c * _BOOT_CHUNK : (c + 1) * _BOOT_CHUNK]])
            for c in range((len(keys) + _BOOT_CHUNK - 1) // _BOOT_CHUNK)
        }
        out = pool.map_keyed(_bootstrap_chunk, chunks)
        flat = [v for c in sorted(out) for v in out[c]]
        return dict(zip(keys, flat, strict=True))

    return rmap


def _defender_team_metric(ctx) -> pd.DataFrame:
    """Every scored match's shard, concatenated in sorted key order (legacy `:666-669`). Rebuilt where it
    is needed rather than persisted: a re-serialised concat could coerce a mixed-type id column."""
    parts = _done_results(ctx, "score")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=_EMITTED_SHARD_COLUMNS)


def _bootstrap_work(key, ctx) -> dict:
    """Tier-1 -> gate -> (iff Tier-1 clears) Tier-2 (legacy `:670-675`), the Tier-2 replicates on the pool."""
    import tempfile

    defender_team_metric = _defender_team_metric(ctx)
    tier1 = tier1_counts(defender_team_metric, min_passes_faced=MIN_PASSES_FACED)
    tier2: dict | None = None
    if tier1["n_multi_team_defenders"] >= MIN_MULTI_TEAM_DEFENDERS and not defender_team_metric.empty:
        with tempfile.TemporaryDirectory() as tmp:
            pool = ctx.pool
            rmap = _replicate_map(pool, Path(tmp) / "boot") if pool is not None and pool.workers > 1 else None
            tier2 = tier2_icc(defender_team_metric, replicate_map=rmap)
    return {"tier1": tier1, "tier2": tier2}


def _report_work(key, ctx):
    """The verdict, the ranking and census.json (legacy `:677-713`); decision P4: the corpus-scoped score
    counters, never the pass-scoped `res.manifest()`."""
    from scripts._provenance import git_provenance  # noqa: F401 -- the provenance gate's source check (ADR-037)

    census = ctx.upstream("bootstrap")
    defender_team_metric = _defender_team_metric(ctx)
    verdict = census_gate(census)
    ranking = build_ranking(
        defender_team_metric, licensed=verdict["ranking_licensed"], min_passes_faced=MIN_PASSES_FACED
    )
    refs = ctx.upstream("refs")
    summary = ctx.summary("score")
    prov = ctx.prov
    out = {
        "n_fit_matches": len(refs["fit_refs"]),
        "n_scored_matches": len(refs["score_refs"]),
        "census": census,
        "verdict": verdict,
        "ranking_licensed": verdict["ranking_licensed"],
        "n_ranked": (0 if ranking is None else len(ranking)),
        "constants": {
            "MIN_MULTI_TEAM_DEFENDERS": MIN_MULTI_TEAM_DEFENDERS,
            "MIN_PASSES_FACED": MIN_PASSES_FACED,
            "ICC_LOWER_FLOOR": ICC_LOWER_FLOOR,
            "POWER_FLOOR": POWER_FLOOR,
            "ICC_EFFECT_SIZE": ICC_EFFECT_SIZE,
        },
        "honest_limit": (
            "A NEGATIVE verdict is EXPECTED on a single-club / national-team corpus: one player = one "
            "team, so the defender-vs-team confound is unidentifiable and a per-defender ranking is NOT "
            "licensed. The census measures this rather than assuming it; a multi-club transfer corpus is "
            "what supplies the cross-team observations a ranking needs."
        ),
        # Decision P4 (ADR-128): CORPUS-scoped counters (the values a single fresh serial pass reports).
        "generation": ctx.tokens["score"],
        "n_attempted": summary["n_units"],
        "n_failed": summary["n_failed"],
        "n_counters_unrecorded": 0,
        "n_excluded": summary["n_excluded"],
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
        "input_contract": input_contract(),
    }
    files = {"census.json": json.dumps(out, indent=2, default=str).encode("utf-8")}
    if ranking is not None and not ranking.empty:
        buf = io.BytesIO()
        ranking.to_parquet(buf, index=False)
        files["ranking.parquet"] = buf.getvalue()
    print(json.dumps({k: v for k, v in out.items() if k != "input_contract"}, indent=2, default=str))
    return FileSet(files)


_CORPUS_PARAMS = (
    "all_competitions",
    "competitions_json",
    "competition_id",
    "season_id",
    "max_matches",
    "fit_fraction",
    "match_ids_json",
)

PIPELINE = Pipeline(
    name="territory_ranking_census",
    module="scripts.build_territory_ranking_census",
    params=_PARAMS,
    stages=(
        Stage(
            name="refs",
            units=lambda ctx: ["only"],
            work=_refs_work,
            reduce=lambda r, ctx: r["only"],
            params=_CORPUS_PARAMS,
            declare=_corpus_declare,
            uses_dataset=True,
            dataset_files=lambda ctx: (),  # open data over HTTP: no dataset file
        ),
        ItemStage(
            name="fit_prepass",
            refs=lambda ctx: _refs_of(ctx, "fit_refs"),
            key=lambda ref: ref.key,
            load=_load,
            work=lambda item, ctx: _prune_actions_slice(item[2]),
            needs=("refs",),
            declare=_fit_declare,
            cross_commit_version="census-fit-1",
            uses_dataset=True,
            max_failures=None,  # the legacy for_each recorded failures and fit over the rest
        ),
        Stage(
            name="prefit",
            units=lambda ctx: ["only"],
            work=_prefit_work,
            needs=("fit_prepass",),
            reduce=lambda r, ctx: r["only"],
        ),
        ItemStage(
            name="score",
            refs=lambda ctx: _refs_of(ctx, "score_refs"),
            key=lambda ref: str(ref.match_id),  # the legacy key: the score shard filenames are unchanged
            load=_load,
            work=_score_work,
            needs=("refs", "prefit"),
            declare=_score_declare,
            cross_commit_version="census-score-1",
            uses_dataset=True,
            max_failures=None,  # the legacy for_each recorded failures and censused the rest
        ),
        Stage(
            name="bootstrap",
            units=lambda ctx: ["only"],
            work=_bootstrap_work,
            needs=("score",),
            reduce=lambda r, ctx: r["only"],
            inner_parallel=True,
            inner_width=8,
        ),
        Stage(
            name="report",
            units=lambda ctx: ["only"],
            work=_report_work,
            needs=("refs", "score", "bootstrap"),
            reduce=lambda r, ctx: r["only"],
        ),
    ),
    output="report",
)

if __name__ == "__main__":
    raise SystemExit(main(PIPELINE))
