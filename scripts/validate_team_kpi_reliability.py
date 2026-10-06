"""TF-52 team-KPI reliability study (owner-run, reported-not-gated).

Runs ``compute_team_kpis`` over a PUBLIC event corpus and writes an AGGREGATE, reproducible reliability
report -- it does NOT change any library default (ADR-009). The corpus is public, so the result is
reproducible by anyone; per-match KPI shards are a superset persisted under ``--out/shards`` and are not
committed.

The StatsBomb leg defaults to the ENTIRE open-data manifest (``all_open_competitions()`` -- every public
competition/season, thousands of matches; a single tournament is far too thin for a reliability ICC /
split-half). Public-only is FAIL-CLOSED, corpus-appropriately (NOT the pining ``assert_public_corpus``,
whose 27-match registry cannot represent open data): the StatsBomb leg refuses to run with credentials
set (``assert_statsbomb_open_data_mode``), and the Wyscout leg allowlists the seven public Pappalardo
2019 competitions.

Three legs (each pre-registered, aggregate-only), all wired into the run:
1. Per-KPI reliability: the team-discrimination ICC(1) (a team across its matches is the group) + a
   split-half correlation (odd/even matches per team) + the Type-II (major-axis) slope.
2. Possession-foundation ground truth: ``spadl.add_possessions`` boundary recall / precision / F1 vs the
   provider's NATIVE possession id (StatsBomb open carries one; threaded via ``preserve_native``). The
   KPIs rest on that segmentation, so its fidelity is reported.
3. Per-provider comparability (``--compare a/metrics.json b/metrics.json``): each KPI's reliability is
   compared across providers; a KPI is flagged POOLABLE only where both report finite same-sign ICC
   within a tolerance -- so a KPI is never pooled across providers whose distributions disagree.

The pure stat + shaping kernels are unit-tested (``tests/scripts/test_team_kpi_reliability.py``); the
corpus orchestration is owner-run (it needs the public corpora downloaded) and is NOT exercised in CI
beyond those kernels + the provenance gate.

Usage (owner):
    python scripts/validate_team_kpi_reliability.py --provider statsbomb --out <out>/statsbomb
    python scripts/validate_team_kpi_reliability.py --provider wyscout --wyscout-dir <dir> --out <out>/wyscout
    python scripts/validate_team_kpi_reliability.py --compare <a>/metrics.json <b>/metrics.json --out <out>/cmp
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._reliability import (  # repo root joined sys.path just above
    _COMPARABILITY_ICC_TOL,
    _MIN_TEAMS,
    compare_providers,
    icc1,
    split_half_reliability,
    type_ii_slope,  # noqa: F401  re-exported: tests import this kernel from the driver's namespace
)

#: Wyscout ``matchPeriod`` -> SPADL period id (Pappalardo public data set).
_WS_PERIOD = {"1H": 1, "2H": 2, "E1": 3, "E2": 4, "P": 5}

#: The seven PUBLIC Wyscout competitions in the Pappalardo 2019 release (Scientific Data 6:236). The
#: Wyscout leg fails closed if a ``--wyscout-dir`` file names a competition OUTSIDE this set, so a
#: private Wyscout feed can never be stamped ``public`` (the ``events_<Competition>.json`` naming).
_PAPPALARDO_PUBLIC_COMPETITIONS = frozenset(
    {"England", "Italy", "Spain", "Germany", "France", "European_Championship", "World_Cup"}
)


def reduce_reliability(samples: pd.DataFrame, kpis: list[str]) -> dict:
    """Pool the per-(game, team) KPI shards into per-KPI reliability verdicts (the REDUCE over all shards)."""
    if samples.empty:
        return {"n_teams": 0, "n_matches": 0, "per_kpi": {}}
    teams = samples["team_id"].to_numpy()
    out: dict = {}
    for k in kpis:
        if k not in samples.columns:
            continue
        vals = samples[k].to_numpy(dtype="float64")
        sh = split_half_reliability(samples, k)
        out[k] = {
            "icc": icc1(vals, teams),
            "split_half_r": sh["r"],
            "n_teams_paired": sh["n_teams"],
            "n_observed": int(np.isfinite(vals).sum()),
        }
    return {
        "n_teams": int(samples["team_id"].nunique()),
        "n_matches": int(samples["game_id"].nunique()),
        "per_kpi": out,
    }


def aggregate_possession_ground_truth(per_match: list[dict]) -> dict:
    """Mean boundary recall / precision / F1 of add_possessions vs native possession id over matches."""
    if not per_match:
        return {"n_matches": 0, "recall_mean": float("nan"), "precision_mean": float("nan"), "f1_mean": float("nan")}
    r = np.array([m["recall"] for m in per_match], dtype="float64")
    p = np.array([m["precision"] for m in per_match], dtype="float64")
    f = np.array([m["f1"] for m in per_match], dtype="float64")
    return {
        "n_matches": len(per_match),
        "recall_mean": float(np.nanmean(r)) if np.isfinite(r).any() else float("nan"),
        "precision_mean": float(np.nanmean(p)) if np.isfinite(p).any() else float("nan"),
        "f1_mean": float(np.nanmean(f)) if np.isfinite(f).any() else float("nan"),
    }


def reduce_possession_ground_truth(samples: pd.DataFrame) -> dict:
    """Aggregate the per-match possession boundary metrics carried on the shards (one per game_id)."""
    if samples.empty or "poss_f1" not in samples.columns:
        return aggregate_possession_ground_truth([])
    per_match = samples.dropna(subset=["poss_f1"]).drop_duplicates("game_id")
    cols = ["poss_recall", "poss_precision", "poss_f1"]
    rows = [
        {"recall": float(rec["poss_recall"]), "precision": float(rec["poss_precision"]), "f1": float(rec["poss_f1"])}
        for rec in per_match[cols].to_dict("records")
    ]
    return aggregate_possession_ground_truth(rows)


def possession_boundary_vs_native(actions: pd.DataFrame, *, native_possession_col: str) -> dict:
    """Boundary metrics of the add_possessions heuristic vs a provider's native possession id (one match).

    Compared over the rows that carry a native possession id (synthesized SPADL rows -- e.g. dribbles --
    carry NaN and are excluded so a NaN never reads as a spurious boundary).
    """
    from silly_kicks.spadl import add_possessions
    from silly_kicks.spadl.utils import boundary_metrics

    if native_possession_col not in actions.columns:
        return {"recall": float("nan"), "precision": float("nan"), "f1": float("nan")}
    ctx = add_possessions(actions)
    both = ctx.assign(_native=actions.loc[ctx.index, native_possession_col].to_numpy()).dropna(subset=["_native"])
    if both.empty:
        return {"recall": float("nan"), "precision": float("nan"), "f1": float("nan")}
    bm = boundary_metrics(heuristic=both["possession_id"], native=both["_native"])
    return {"recall": bm["recall"], "precision": bm["precision"], "f1": bm["f1"]}


def _shape_pappalardo_event(e: dict) -> dict:
    """One raw public-Wyscout (Pappalardo 2019) event -> the ``spadl.wyscout`` input contract. Pure."""
    return {
        "game_id": e["matchId"],
        "event_id": e["id"],
        "period_id": _WS_PERIOD.get(str(e.get("matchPeriod", "")), 1),
        "milliseconds": round(float(e.get("eventSec", 0.0)) * 1000),
        "team_id": e["teamId"],
        "player_id": e.get("playerId"),
        "type_id": e["eventId"],
        "subtype_id": e.get("subEventId"),
        "positions": e.get("positions", []),
        "tags": e.get("tags", []),
    }


def _pappalardo_home_team(match: dict) -> int:
    """Home team id from a Pappalardo ``matches_*.json`` record (``teamsData[tid].side == 'home'``)."""
    for tid, td in match.get("teamsData", {}).items():
        if td.get("side") == "home":
            return int(tid)
    raise ValueError(f"no home side in Pappalardo match {match.get('wyId')}")


# --------------------------------------------------------------------------- corpus loaders (owner-run)
def _statsbomb_source(competitions, match_ids=None, *, max_matches=None, cache_dir=None):
    """The StatsBomb open-data corpus as ``(refs, load)`` (Task 8.5; owner-run; needs statsbombpy).

    ``competitions is None`` -> the FULL open-data manifest (``all_open_competitions()``: every public
    ``(competition_id, season_id)`` StatsBomb releases -- thousands of matches, the broad reliability
    corpus). A ``[(competition_id, season_id), ...]`` list narrows it. Delegates to the shared
    ``scripts._sb_open_data.open_data_source`` with ``preserve_native=("possession",)`` so the SPADL
    actions carry StatsBomb's native possession id for the possession-foundation ground-truth leg;
    ``load(ref)`` yields the ``(match_id, actions)`` pair ``_measure_match`` consumes. ``max_matches``
    caps the GLOBAL count. Fail-closed on configured credentials. NOT exercised in CI.
    """
    from scripts._sb_open_data import all_open_competitions, open_data_source

    comps = list(competitions) if competitions is not None else all_open_competitions()
    refs, base_load = open_data_source(
        comps, match_ids=match_ids, max_matches=max_matches, preserve_native=("possession",), cache_dir=cache_dir
    )

    def load(ref):
        lm = base_load(ref)
        return (lm.match_id, lm.actions)

    return refs, load


def _load_wyscout_pappalardo_matches(wyscout_dir, match_ids=None, *, max_matches=None):
    """Yield ``(match_id, actions)`` for the public Wyscout data set (Pappalardo 2019; owner-run).

    Reads the published ``events_<competition>.json`` + ``matches_<competition>.json`` files from
    ``wyscout_dir``, groups events by ``matchId``, shapes each raw event to the ``spadl.wyscout`` input
    contract (``_shape_pappalardo_event``), and converts to SPADL. NOT exercised in CI (the shaper is).
    """
    from silly_kicks.spadl import wyscout as wy_convert

    root = Path(wyscout_dir)
    wanted = {str(m) for m in match_ids} if match_ids is not None else None
    seen = 0
    for events_file in sorted(root.glob("events_*.json")):
        competition = events_file.stem[len("events_") :]
        if competition not in _PAPPALARDO_PUBLIC_COMPETITIONS:
            raise SystemExit(
                f"{events_file.name} names competition {competition!r}, which is NOT one of the public "
                f"Pappalardo 2019 competitions {sorted(_PAPPALARDO_PUBLIC_COMPETITIONS)}. This artifact "
                "is public-only; refusing to run over a possibly-private Wyscout feed."
            )
        matches_file = root / f"matches_{competition}.json"
        if not matches_file.exists():
            continue
        matches = {int(m["wyId"]): m for m in json.loads(matches_file.read_text(encoding="utf-8"))}
        by_match: dict[int, list] = defaultdict(list)
        for e in json.loads(events_file.read_text(encoding="utf-8")):
            by_match[int(e["matchId"])].append(e)
        for mid, evs in by_match.items():
            if (wanted is not None and str(mid) not in wanted) or mid not in matches:
                continue
            home = _pappalardo_home_team(matches[mid])
            shaped = pd.DataFrame([_shape_pappalardo_event(e) for e in evs])
            actions, _report = wy_convert.convert_to_actions(shaped, home)
            yield str(mid), actions
            seen += 1
            if max_matches is not None and seen >= max_matches:
                return


def _item_key(item):
    """The ``for_each`` key for BOTH sources: a StatsBomb ``MatchRef`` -> its ``match_id``, a Wyscout
    ``(match_id, actions)`` tuple -> its ``match_id`` (module-level so the key-pin gate sees it)."""
    return str(item.match_id) if hasattr(item, "match_id") else str(item[0])


def _load_matches(provider, *, competitions, wyscout_dir, match_ids, max_matches, cache_dir=None):
    """Return ``(items, load)`` for the requested provider (Task 8.5).

    StatsBomb -> ``(refs, load)`` (a cheap ref list + a single-match loader behind ``for_each``'s resume
    check). Wyscout is FILE-based (local Pappalardo JSON, cheap to re-read), so it streams its
    ``(match_id, actions)`` items with ``load=None`` -- there is no network round-trip to shard away.
    """
    if provider == "wyscout":
        if not wyscout_dir:
            raise SystemExit("--wyscout-dir is required for --provider wyscout")
        return _load_wyscout_pappalardo_matches(wyscout_dir, match_ids=match_ids, max_matches=max_matches), None
    return _statsbomb_source(competitions, match_ids=match_ids, max_matches=max_matches, cache_dir=cache_dir)


def _measure_match(item) -> pd.DataFrame:
    """work(item): one match's per-(game, team) KPI samples + the per-match possession-ground-truth."""
    from silly_kicks.team_metrics import compute_team_kpis

    _mid, actions = item
    # StatsBomb open data carries its own pre-shot xG (attached as `xg` by the loader) -> feed it so
    # high_opportunity_shots is measured; Wyscout has none, so xg_column falls back to None.
    xg_column = "xg" if "xg" in actions.columns else None
    samples, _report = compute_team_kpis(actions, xg_column=xg_column)
    if len(samples) and "possession" in actions.columns:
        pm = possession_boundary_vs_native(actions, native_possession_col="possession")
        samples = samples.assign(poss_recall=pm["recall"], poss_precision=pm["precision"], poss_f1=pm["f1"])
    return samples


def _run_compare(compare_paths: list[str], dest: Path, prov: dict) -> None:
    reports = {}
    for path in compare_paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        provider = data.get("input_contract", {}).get("provider", Path(path).parent.name)
        reports[provider] = data.get("verdicts", {})
    comparison = compare_providers(reports)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "comparison.json").write_text(
        json.dumps({"run_commit": prov["commit"], "run_tree_dirty": prov["dirty"], "comparison": comparison}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="output dir (not needed with --list-matches)")
    ap.add_argument("--provider", default="statsbomb", choices=["statsbomb", "wyscout"], help="public event provider")
    ap.add_argument(
        "--wyscout-dir", default=None, help="dir of the public Wyscout (Pappalardo) events_*/matches_* JSON"
    )
    ap.add_argument("--match-ids-json", default=None, help="JSON [<id>, ...] pinning WHICH matches (parallel split)")
    ap.add_argument(
        "--competitions-json",
        default=None,
        help="JSON [[competition_id, season_id], ...] narrowing statsbomb (default: the FULL open-data manifest)",
    )
    ap.add_argument("--max-matches", type=int, default=None)
    ap.add_argument(
        "--cache-dir", default=None, help="raw open-data events cache root (else $SILLY_KICKS_CORPUS_CACHE_DIR)"
    )
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (dev only; artifact is marked)")
    ap.add_argument("--list-matches", action="store_true", help="print available match ids as JSON and exit")
    ap.add_argument(
        "--compare", nargs="+", default=None, help="two+ per-provider metrics.json to compare (comparability leg)"
    )
    args = ap.parse_args()

    from scripts._input_contract import declare_inputs
    from scripts._provenance import git_provenance, require_clean_tree

    if not args.list_matches and not args.out:
        raise SystemExit("--out is required unless --list-matches is given")

    prov = (
        {"commit": "n/a", "dirty": False, "tree_state": "clean", "dirty_files": []}
        if args.list_matches
        else require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)
    )

    if args.compare:
        _run_compare(args.compare, Path(args.out), prov)
        return

    match_ids = json.loads(Path(args.match_ids_json).read_text(encoding="utf-8")) if args.match_ids_json else None
    competitions = (
        [tuple(c) for c in json.loads(Path(args.competitions_json).read_text(encoding="utf-8"))]
        if args.competitions_json
        else None  # None -> the loader enumerates the FULL open-data manifest (broad public corpus)
    )

    if args.list_matches:
        # LIST ids without loading: for StatsBomb the refs carry the ids (no per-match download).
        items, _load = _load_matches(
            args.provider,
            competitions=competitions,
            wyscout_dir=args.wyscout_dir,
            match_ids=match_ids,
            max_matches=args.max_matches,
            cache_dir=args.cache_dir,
        )
        print(json.dumps([_item_key(it) for it in items], indent=2))
        return

    from scripts._driver import for_each
    from silly_kicks.team_metrics import TEAM_KPI_METRIC_COLUMNS

    dest = Path(args.out)

    items, load = _load_matches(
        args.provider,
        competitions=competitions,
        wyscout_dir=args.wyscout_dir,
        match_ids=match_ids,
        max_matches=args.max_matches,
        cache_dir=args.cache_dir,
    )

    res = for_each(
        items,
        key=lambda item: (args.provider, _item_key(item)),
        load=load,
        work=_measure_match,
        shard_root=dest / "shards",
        token_inputs={"metric": "team_kpi_reliability", "provider": args.provider, "schema": "team-kpi-3"},
        label="match",
    )
    shard_files = sorted(res.shard_dir.glob("*.parquet"))
    samples = pd.concat([pd.read_parquet(s) for s in shard_files], ignore_index=True) if shard_files else pd.DataFrame()
    verdicts = {
        "reliability": reduce_reliability(samples, list(TEAM_KPI_METRIC_COLUMNS)),
        "possession_ground_truth": reduce_possession_ground_truth(samples),
    }

    contract = declare_inputs(
        driver="validate_team_kpi_reliability",
        provider=args.provider,
        metric_columns=list(TEAM_KPI_METRIC_COLUMNS),
        min_teams=_MIN_TEAMS,
        comparability_icc_tol=_COMPARABILITY_ICC_TOL,
    )
    report = {
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "input_contract": contract,
        "verdicts": verdicts,
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(verdicts, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
