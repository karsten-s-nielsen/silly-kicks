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
with NO network in ``tests/scripts/test_build_territory_ranking_census.py``; ``main()`` is the I/O:
clean-tree guard FIRST (ADR-037), fail-closed public-only StatsBomb open data
(``assert_statsbomb_open_data_mode``), ``for_each``-sharded per-match counterfactual compute (ADR-052)
with an ``xt`` + ``PassCompletionModel`` fit INLINE on a LEAKAGE-DISJOINT fit split (self-contained --
no forward dependency on the validator), and an input contract (ADR-056). ``statsbombpy`` is an
optional ``scripts/`` network dep for the public lineups (defender -> team identity).

HONEST LIMIT (recorded in the artifact): a NEGATIVE verdict is the expected outcome on a single-club /
national-team corpus, where one player = one team and the defender-vs-team confound is unidentifiable.
The census exists to make that verdict MEASURED rather than assumed -- and to license the ranking on
the day a multi-club transfer corpus supplies the cross-team observations it needs.

The census aggregates the ENTIRE shard generation for its ``--out`` directory (a whole-corpus
census): the corpus selectors (``--competition-id`` / ``--season-id`` / ``--max-matches``) are
deliberately NOT in the ``for_each`` ``token_inputs``, so the generation is a SUPERSET of any one
narrowed run (ADR-052 reconcile-superset behavior -- intended). Running a narrowed selector (e.g.
``--max-matches 10``) and later the full corpus into the SAME ``--out`` therefore combines the
superset (stale narrow shards + full); an owner wanting a clean census of a narrowed corpus should
use a FRESH ``--out``.

Usage (owner):
    python scripts/build_territory_ranking_census.py --out docs/research/territory_ranking_census
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from _crossed_icc import bootstrap_icc_and_power  # scripts/ on sys.path (conftest)

from scripts._input_contract import declare_inputs
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


def tier2_icc(defender_team_metric: pd.DataFrame) -> dict:
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

    _provider, match_id, actions, _frames, _home = item
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


def _corpus_matches(competitions, *, match_ids, max_matches):
    """Chain every ``(competition_id, season_id)``'s open-data matches into ONE ``(match_id, actions)``
    stream (mirrors ``validate_territory_counterfactual._corpus_matches``).

    ``competitions`` is ``all_open_competitions()`` (the FULL public open manifest) under
    ``--all-competitions`` or the ``--competitions-json`` override; the counterfactual metric is
    event-only, so only ``(match_id, actions)`` is kept. ``max_matches`` caps the GLOBAL count across
    every competition (a broad census wants the whole corpus by default).
    """
    from scripts._sb_open_data import load_open_data_matches

    seen = 0
    for competition_id, season_id in competitions:
        for _provider, match_id, actions, _frames, _home in load_open_data_matches(
            competition_id=competition_id,
            season_id=season_id,
            match_ids=(match_ids or {}).get("statsbomb"),
        ):
            yield str(match_id), actions
            seen += 1
            if max_matches is not None and seen >= max_matches:
                return


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="output dir (not needed with --list-matches)")
    # Single-competition selectors default to None sentinels so an EXPLICIT --competition-id/--season-id
    # (which conflicts with the broad-corpus flags) is distinguishable from the resolved 43/106 default.
    ap.add_argument("--competition-id", type=int, default=None, help="open-data competition (default 43 = World Cup)")
    ap.add_argument("--season-id", type=int, default=None, help="open-data season (default 106 = 2022)")
    ap.add_argument(
        "--all-competitions",
        action="store_true",
        help="census the FULL public open-data corpus (all_open_competitions()). REQUIRED for a crossed "
        "defender+team ICC ranking -- defenders on >=2 teams only exist ACROSS competitions/seasons, so a "
        "single-competition census can only ever return 'not licensed'. Mutually exclusive with "
        "--competition-id/--season-id.",
    )
    ap.add_argument(
        "--competitions-json",
        default=None,
        help="JSON [[competition_id, season_id], ...] narrowing the broad corpus to a chosen list "
        "(mirrors validate_territory_counterfactual). Mutually exclusive with --competition-id/--season-id.",
    )
    ap.add_argument("--max-matches", type=int, default=None)
    ap.add_argument(
        "--fit-fraction",
        type=float,
        default=0.5,
        help="fraction of matches HELD OUT to fit xt/completion (leakage-disjoint); the rest are scored.",
    )
    ap.add_argument(
        "--match-ids-json", default=None, help='JSON ["3857276", ...] pinning WHICH matches (parallel split).'
    )
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (dev only; artifact is marked)")
    ap.add_argument("--list-matches", action="store_true", help="print available scored match ids as JSON and exit")
    args = ap.parse_args()

    broad_corpus = args.all_competitions or args.competitions_json is not None
    if broad_corpus and (args.competition_id is not None or args.season_id is not None):
        ap.error(
            "--all-competitions / --competitions-json census the whole open-data corpus and cannot be "
            "combined with the single-competition --competition-id / --season-id selectors."
        )

    from scripts._provenance import git_provenance, require_clean_tree
    from scripts._sb_open_data import (
        all_open_competitions,
        assert_statsbomb_open_data_mode,
        load_open_data_matches,
    )

    if not args.list_matches and not args.out:
        raise SystemExit("--out is required unless --list-matches is given")

    # Clean-tree guard FIRST, before any corpus work (ADR-037). --list-matches writes no artifact.
    prov = (
        {"commit": "n/a", "dirty": False, "tree_state": "clean", "dirty_files": []}
        if args.list_matches
        else require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)
    )
    assert_statsbomb_open_data_mode()  # fail-closed: never pull the private API for a public artifact

    match_ids = json.loads(Path(args.match_ids_json).read_text(encoding="utf-8")) if args.match_ids_json else None

    # Resolve the corpus. BROAD mode (--all-competitions / --competitions-json) chains EVERY
    # (competition, season)'s matches so the ONE leakage-disjoint xt/completion fit below pools across
    # the whole corpus (a crossed defender+team ICC needs defenders on >=2 teams, which only exist
    # ACROSS competitions -- a single competition can only ever return "not licensed"). SINGLE mode
    # (the default) keeps the historical WC2022 path, byte-identical.
    if broad_corpus:
        competitions = (
            [tuple(c) for c in json.loads(Path(args.competitions_json).read_text(encoding="utf-8"))]
            if args.competitions_json
            else all_open_competitions()
        )
        corpus_source = _corpus_matches(competitions, match_ids=match_ids, max_matches=args.max_matches)
    else:
        competition_id = 43 if args.competition_id is None else args.competition_id
        season_id = 106 if args.season_id is None else args.season_id
        corpus_source = (
            (str(match_id), actions)
            for _provider, match_id, actions, _frames, _home in load_open_data_matches(
                competition_id=competition_id,
                season_id=season_id,
                match_ids=match_ids,
                max_matches=args.max_matches,
            )
        )

    # Stream the corpus ONCE, keeping compact per-match (id, actions) -- the counterfactual metric is
    # event-only (frames unused). Materialized so the leakage-disjoint split is deterministic. An
    # explicit per-match loop (not a comprehension) so the ADR-052 corpus-driver resilience gate sees
    # this as a corpus walker and asserts the for_each adoption below.
    corpus: list[tuple[str, pd.DataFrame]] = []
    for match_id, actions in corpus_source:
        corpus.append((str(match_id), actions))
    # Leakage-disjoint split: the first `fit_fraction` of the (sorted) matches fit xt/completion; the
    # rest are scored. Deterministic (sorted by id) so a resume/partition sees the same split.
    corpus.sort(key=lambda mi: mi[0])
    n_fit = max(1, round(len(corpus) * args.fit_fraction)) if corpus else 0
    fit_matches, score_matches = corpus[:n_fit], corpus[n_fit:]

    if args.list_matches:
        print(json.dumps([mid for mid, _ in score_matches], indent=2))
        return

    from scripts._driver import for_each

    dest = Path(args.out)
    xt, completion_model = _fit_disjoint_models([a for _mid, a in fit_matches])
    score_by_id = dict(score_matches)

    def _work(mid: str) -> pd.DataFrame:
        actions = score_by_id[mid]
        return _score_match(("statsbomb", mid, actions, pd.DataFrame(), None), xt=xt, completion_model=completion_model)

    # The shard-generation token. The single-competition (default) path keeps its historical token
    # BYTE-IDENTICAL, so its generation directory is unchanged. A broad-corpus run adds a "source"
    # marker (the 4.77.1 stale-shard rule) so its generation digest cannot collide with the
    # single-competition one, and a --competitions-json narrowing keys on its explicit competition list
    # so two narrow runs also disjoin.
    token_inputs: dict = {
        "metric": "territory_ranking_census",
        "schema": _SHARD_SCHEMA_VERSION,
        "xt": "leakage_disjoint_fit",
        "fit_fraction": args.fit_fraction,
    }
    if broad_corpus:
        token_inputs["source"] = (
            "open-data-all"
            if not args.competitions_json
            else {"open-data-narrowed": sorted(tuple(c) for c in competitions)}
        )

    res = for_each(
        list(score_by_id.keys()),
        key=lambda mid: str(mid),
        work=_work,
        shard_root=dest / "shards",
        token_inputs=token_inputs,
        label="match",
    )

    # Combine the ENTIRE shard generation for this --out dir (a whole-corpus census). The corpus
    # selectors (--competition-id/--season-id/--max-matches) are deliberately NOT in token_inputs, so
    # the generation directory is a SUPERSET of any one narrowed run (ADR-052 reconcile-superset
    # behavior, intended). CONSEQUENCE: a narrowed run (e.g. --max-matches 10) and a later full-corpus
    # run into the SAME --out combine together (stale narrow shards + full). For a clean census of a
    # narrowed corpus, use a FRESH --out.
    shards = [pd.read_parquet(s) for s in sorted(res.shard_dir.glob("*.parquet"))]
    defender_team_metric = (
        pd.concat(shards, ignore_index=True) if shards else pd.DataFrame(columns=_EMITTED_SHARD_COLUMNS)
    )

    # Tier-1 -> gate -> (iff Tier-1 clears) Tier-2 -> gate.
    tier1 = tier1_counts(defender_team_metric, min_passes_faced=MIN_PASSES_FACED)
    tier2: dict | None = None
    if tier1["n_multi_team_defenders"] >= MIN_MULTI_TEAM_DEFENDERS and not defender_team_metric.empty:
        tier2 = tier2_icc(defender_team_metric)
    census = {"tier1": tier1, "tier2": tier2}
    verdict = census_gate(census)

    ranking = build_ranking(
        defender_team_metric, licensed=verdict["ranking_licensed"], min_passes_faced=MIN_PASSES_FACED
    )

    out = {
        "n_fit_matches": len(fit_matches),
        "n_scored_matches": len(score_matches),
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
        **res.manifest(),
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
        "input_contract": input_contract(),
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "census.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    if ranking is not None and not ranking.empty:
        ranking.to_parquet(dest / "ranking.parquet", index=False)
    print(json.dumps({k: v for k, v in out.items() if k != "input_contract"}, indent=2, default=str))


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
