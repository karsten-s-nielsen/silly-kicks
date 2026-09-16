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
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_MIN_TEAMS = 3  # a correlation / ICC below this team count is not reported
_COMPARABILITY_ICC_TOL = 0.20  # max cross-provider ICC spread for a KPI to be flagged poolable

#: Wyscout ``matchPeriod`` -> SPADL period id (Pappalardo public data set).
_WS_PERIOD = {"1H": 1, "2H": 2, "E1": 3, "E2": 4, "P": 5}

#: The seven PUBLIC Wyscout competitions in the Pappalardo 2019 release (Scientific Data 6:236). The
#: Wyscout leg fails closed if a ``--wyscout-dir`` file names a competition OUTSIDE this set, so a
#: private Wyscout feed can never be stamped ``public`` (the ``events_<Competition>.json`` naming).
_PAPPALARDO_PUBLIC_COMPETITIONS = frozenset(
    {"England", "Italy", "Spain", "Germany", "France", "European_Championship", "World_Cup"}
)


# --------------------------------------------------------------------------- pure stat kernels
def icc1(values: np.ndarray, groups: np.ndarray) -> float:
    """One-way random-effects ICC(1): between-group / total variance (team-discrimination). Pure numpy."""
    df = pd.DataFrame({"v": np.asarray(values, dtype="float64"), "g": groups}).dropna(subset=["v"])
    k = df["g"].nunique()
    n = len(df)
    if k < 2 or n <= k:
        return float("nan")
    grand = df["v"].mean()
    gm = df.groupby("g")["v"]
    ni = gm.count().to_numpy(dtype="float64")
    mi = gm.mean().to_numpy()
    ssb = float(np.sum(ni * (mi - grand) ** 2))
    ssw = float(np.sum((df["v"].to_numpy() - df.groupby("g")["v"].transform("mean").to_numpy()) ** 2))
    msb = ssb / (k - 1)
    msw = ssw / (n - k)
    n0 = (n - np.sum(ni**2) / n) / (k - 1)
    denom = msb + (n0 - 1) * msw
    return (msb - msw) / denom if denom > 0 else float("nan")


def _stable_half(game_id) -> int:
    """Deterministic 0/1 split of a match id (stable across runs / platforms)."""
    return int(hashlib.sha256(str(game_id).encode("utf-8")).hexdigest(), 16) % 2


def split_half_reliability(samples: pd.DataFrame, kpi: str, *, team_col="team_id", id_col="game_id") -> dict:
    """Split each team's matches odd/even, mean the KPI per half, Pearson r of the two halves across teams."""
    df = samples[[team_col, id_col, kpi]].dropna()
    if df.empty:
        return {"r": float("nan"), "n_teams": 0}
    df = df.assign(_h=df[id_col].map(_stable_half))
    means = df.groupby([team_col, "_h"])[kpi].mean().unstack("_h")
    if 0 not in means.columns or 1 not in means.columns:
        return {"r": float("nan"), "n_teams": 0}
    pair = means.dropna(subset=[0, 1])
    if len(pair) < _MIN_TEAMS or pair[0].std() == 0 or pair[1].std() == 0:
        return {"r": float("nan"), "n_teams": len(pair)}
    r = float(np.corrcoef(pair[0].to_numpy(), pair[1].to_numpy())[0, 1])
    return {"r": r, "n_teams": len(pair)}


def type_ii_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Major-axis (orthogonal) regression slope = sign(corr) * sd(y) / sd(x)."""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < _MIN_TEAMS or x.std() == 0 or y.std() == 0:
        return float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    return float(np.sign(r) * y.std() / x.std())


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


def compare_providers(reports: dict[str, dict]) -> dict:
    """Per-KPI cross-provider comparability: POOLABLE iff every provider reports finite same-sign ICC
    within ``_COMPARABILITY_ICC_TOL``. ``reports`` maps provider -> its ``verdicts`` dict."""
    providers = list(reports)
    if len(providers) < 2:
        return {"n_providers": len(providers), "per_kpi": {}}
    kpis: set[str] = set()
    for r in reports.values():
        kpis |= set(r.get("reliability", {}).get("per_kpi", {}))
    out: dict = {}
    for k in sorted(kpis):
        vals = {p: reports[p].get("reliability", {}).get("per_kpi", {}).get(k, {}) for p in providers}
        iccs = [v.get("icc") for v in vals.values()]
        finite = [x for x in iccs if x is not None and np.isfinite(x)]
        poolable = (
            len(finite) == len(providers)
            and all(np.sign(finite[0]) == np.sign(x) for x in finite)
            and (max(finite) - min(finite)) <= _COMPARABILITY_ICC_TOL
        )
        out[k] = {
            "providers": {
                p: {"icc": vals[p].get("icc"), "split_half_r": vals[p].get("split_half_r")} for p in providers
            },
            "icc_spread": float(max(finite) - min(finite)) if len(finite) == len(providers) else None,
            "poolable": bool(poolable),
        }
    return {"n_providers": len(providers), "per_kpi": out}


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
def _load_statsbomb_open_matches(competitions, match_ids=None, *, max_matches=None):
    """Yield ``(match_id, actions)`` for StatsBomb open-data matches (owner-run; needs statsbombpy).

    ``competitions is None`` -> the FULL open-data manifest (``all_open_competitions()``: every public
    ``(competition_id, season_id)`` StatsBomb releases -- thousands of matches, the broad reliability
    corpus). A ``[(competition_id, season_id), ...]`` list narrows it. Delegates to the shared
    ``scripts._sb_open_data.load_open_data_matches`` (the ONE pyright-clean SB-open loader) with
    ``preserve_native=("possession",)`` so the SPADL actions carry StatsBomb's native possession id for
    the possession-foundation ground-truth leg. Both seams fail-closed on configured credentials. NOT
    exercised in CI.
    """
    from scripts._sb_open_data import all_open_competitions, load_open_data_matches

    comps = list(competitions) if competitions is not None else all_open_competitions()
    seen = 0
    for competition_id, season_id in comps:
        for _prov, mid, actions, _frames, _home in load_open_data_matches(
            competition_id=competition_id,
            season_id=season_id,
            match_ids=match_ids,
            preserve_native=("possession",),
        ):
            yield mid, actions
            seen += 1
            if max_matches is not None and seen >= max_matches:
                return


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


def _load_matches(provider, *, competitions, wyscout_dir, match_ids, max_matches):
    if provider == "wyscout":
        if not wyscout_dir:
            raise SystemExit("--wyscout-dir is required for --provider wyscout")
        return _load_wyscout_pappalardo_matches(wyscout_dir, match_ids=match_ids, max_matches=max_matches)
    return _load_statsbomb_open_matches(competitions, match_ids=match_ids, max_matches=max_matches)


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
        ids = [
            mid
            for mid, _actions in _load_matches(
                args.provider,
                competitions=competitions,
                wyscout_dir=args.wyscout_dir,
                match_ids=match_ids,
                max_matches=args.max_matches,
            )
        ]
        print(json.dumps(ids, indent=2))
        return

    from scripts._driver import for_each
    from silly_kicks.team_metrics import TEAM_KPI_METRIC_COLUMNS

    dest = Path(args.out)

    def _matches():
        yield from _load_matches(
            args.provider,
            competitions=competitions,
            wyscout_dir=args.wyscout_dir,
            match_ids=match_ids,
            max_matches=args.max_matches,
        )

    res = for_each(
        _matches(),
        key=lambda item: (args.provider, str(item[0])),
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
