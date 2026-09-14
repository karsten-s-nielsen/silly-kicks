"""TF-62 GK build-up decision-quality construct-validity battery (owner-run, reported-not-gated).

Runs the native-tier legs of the spike over a corpus and writes an AGGREGATE report -- it does NOT
change any library default (ADR-009). Per the reversibility-not-provenance rule, aggregate statistics
(ICC / rho / effect sizes) over owner-tier SkillCorner Game-Intelligence data are allowed; the raw data
is never committed. The corpus map is ``for_each`` (ADR-052: per-match shards, resumable, conserving);
the pooled verdicts are CORPUS statistics computed in the REDUCE over ALL shards, NEVER per shard.
Clean-tree guard runs FIRST (ADR-037); the input contract declares which symbols the numbers depend on
(ADR-056).

Legs (native tier; each pre-registered, aggregate-only):
- Responsiveness: ``decision_pct`` vs the random-choice baseline 0.5, and ``decision_value`` vs 0.
- Discrimination: one-way keeper ICC(1) vs a keeper-label permutation null.
- Net of team: leave-one-keeper-out (club-adjusted) + team-fixed-effect residual keeper ICC.
- Transfer: crossing-keeper residual sign-agreement + correlation.

The pure stat kernels (``icc1`` / ``club_adjusted_residuals`` / ``transfer_signs``) are unit-tested
(``tests/scripts/test_gk_decision_battery_kernels.py``); the corpus orchestration is owner-run
(it needs the owner-tier corpus) and is NOT exercised in CI beyond those kernels + the provenance gate.

Usage (owner):
    python scripts/validate_gk_decision.py --out docs/research/gk_decision_construct_validity
"""

from __future__ import annotations

import argparse
import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_RNG = np.random.default_rng(20260912)
_MIN_PER_KEEPER = 5
_DECISION_METRICS = ("decision_value", "sel_efficiency", "decision_pct")


# --------------------------------------------------------------------------- pure stat kernels
def icc1(values: np.ndarray, groups: np.ndarray) -> float:
    """One-way random-effects ICC(1): between-group / total variance. Pure numpy."""
    df = pd.DataFrame({"v": np.asarray(values, dtype="float64"), "g": groups})
    df = df.dropna(subset=["v"])
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


def icc_vs_permutation(df: pd.DataFrame, col: str, *, n_perm: int = 1000) -> dict:
    """Observed keeper ICC(1) + a keeper-label permutation null (p95, p-value)."""
    s = df.dropna(subset=[col]).groupby("keeper").filter(lambda g: len(g) >= _MIN_PER_KEEPER)
    if s["keeper"].nunique() < 2:
        return {
            "icc": float("nan"),
            "null_p95": float("nan"),
            "p": float("nan"),
            "n_keepers": int(s["keeper"].nunique()),
            "n": len(s),
        }
    g_arr = s["keeper"].to_numpy()
    v_arr = s[col].to_numpy(dtype="float64")
    obs = icc1(v_arr, g_arr)
    perm = np.array([icc1(v_arr, _RNG.permutation(g_arr)) for _ in range(n_perm)])
    perm = perm[~np.isnan(perm)]
    return {
        "icc": obs,
        "null_p95": float(np.nanpercentile(perm, 95)) if perm.size else float("nan"),
        "p": float(np.mean(perm >= obs)) if perm.size else float("nan"),
        "n_keepers": int(s["keeper"].nunique()),
        "n": len(s),
    }


def club_adjusted_residuals(df: pd.DataFrame, col: str) -> pd.Series:
    """Net-of-team residual: value minus the LEAVE-ONE-KEEPER-OUT team mean (Eyestone's method).

    Defined only within multi-keeper teams (a single-keeper team has no leave-one-out baseline -> NaN).
    """
    s = df.dropna(subset=[col]).copy()
    team_sum = s.groupby("team")[col].transform("sum")
    team_cnt = s.groupby("team")[col].transform("count")
    ks = s.groupby(["team", "keeper"])[col].transform("sum")
    kc = s.groupby(["team", "keeper"])[col].transform("count")
    denom = team_cnt - kc
    loo = (team_sum - ks) / denom.where(denom > 0)
    return s[col] - loo


def transfer_signs(df: pd.DataFrame, col: str, *, min_per_team: int = _MIN_PER_KEEPER) -> dict:
    """Crossing-keeper transfer test: club-adjusted residual sign-agreement + correlation across a
    keeper's two most-played clubs (>50% / positive => the signal travels with the keeper)."""
    rows = []
    for team, tg in df.dropna(subset=[col]).groupby("team"):
        tsum, tcnt = tg[col].sum(), len(tg)
        for keeper, kg in tg.groupby("keeper"):
            if len(kg) < min_per_team or (tcnt - len(kg)) <= 0:
                continue
            loo = (tsum - kg[col].sum()) / (tcnt - len(kg))
            rows.append({"keeper": keeper, "team": team, "n": len(kg), "resid": kg[col].mean() - loo})
    kt = pd.DataFrame(rows)
    if kt.empty:
        return {"n_transfer_keepers": 0, "sign_agreement": float("nan"), "resid_corr": float("nan")}
    multi = kt.groupby("keeper").filter(lambda g: g["team"].nunique() >= 2)
    pairs = []
    for _keeper, g in multi.groupby("keeper"):
        g2 = g.sort_values("n", ascending=False).head(2)
        if len(g2) == 2:
            pairs.append((float(g2["resid"].iloc[0]), float(g2["resid"].iloc[1])))
    if len(pairs) < 3:
        return {"n_transfer_keepers": len(pairs), "sign_agreement": float("nan"), "resid_corr": float("nan")}
    ra = np.array([p[0] for p in pairs])
    rb = np.array([p[1] for p in pairs])
    from scipy.stats import spearmanr

    res = spearmanr(ra, rb)
    return {
        "n_transfer_keepers": len(pairs),
        "sign_agreement": float(np.mean(np.sign(ra) == np.sign(rb))),
        "resid_corr": float(res.statistic),  # type: ignore[reportAttributeAccessIssue]
    }


def _one_sample_t(values: np.ndarray, mu: float) -> float:
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    sd = v.std(ddof=1)
    return float((v.mean() - mu) / (sd / sqrt(len(v)))) if sd > 0 and len(v) > 1 else float("nan")


def reduce_samples(samples: pd.DataFrame) -> dict:
    """Pool per-decision native samples into the aggregate verdicts (the REDUCE over all shards)."""
    df = samples.rename(columns={"team_id": "team"})
    out: dict = {
        "n_decisions": len(df),
        "n_keepers": int(df["keeper"].nunique()),
        "n_teams": int(df["team"].nunique()),
        "n_crossing_keepers": int((df.groupby("keeper")["team"].nunique() >= 2).sum()),
        "responsiveness": {
            "decision_pct_mean": float(df["decision_pct"].mean()),
            "decision_pct_t_vs_0.5": _one_sample_t(df["decision_pct"].to_numpy(), 0.5),
            "decision_value_mean": float(df["decision_value"].mean()),
            "decision_value_t_vs_0": _one_sample_t(df["decision_value"].to_numpy(), 0.0),
        },
        "discrimination_one_way": {c: icc_vs_permutation(df, c) for c in _DECISION_METRICS},
        "net_of_team": {},
        "transfer": {c: transfer_signs(df, c) for c in _DECISION_METRICS},
    }
    for c in _DECISION_METRICS:
        club = df.assign(resid=club_adjusted_residuals(df, c))
        team_fe = df.assign(resid=df[c] - df.groupby("team")[c].transform("mean"))
        out["net_of_team"][c] = {
            "club_adjusted": icc_vs_permutation(club.dropna(subset=["resid"]), "resid"),
            "team_fixed_effect": icc_vs_permutation(team_fe.dropna(subset=["resid"]), "resid"),
            "team_one_way": icc_vs_permutation(df.rename(columns={"keeper": "_k", "team": "keeper"}), c),
        }
    return out


# --------------------------------------------------------------------------- reconstruction kernels
def fidelity_spearman(recon: pd.DataFrame, native: pd.DataFrame, *, value_col: str, keys: list[str]) -> dict:
    """Rosetta-Stone rank fidelity: reconstructed-from-tracking vs native-GI on the SHARED ``keys``.

    Inner-joins on ``keys`` (a key present on only one side is dropped -- counted in ``n_recon``/
    ``n_native`` vs ``n``), then Spearman rho of ``value_col``. Perfect-agreeing ranking -> rho 1;
    reversed -> -1. This is a RANKING check (what chosen-vs-available needs); it complements, never
    replaces, the option_value magnitude gate (a monotone-biased count preserves rho).
    """
    from scipy.stats import spearmanr

    a = recon[[*keys, value_col]].dropna()
    b = native[[*keys, value_col]].dropna()
    merged = a.merge(b, on=keys, suffixes=("_recon", "_native"))
    n = len(merged)
    if n < 3:
        return {"rho": float("nan"), "p": float("nan"), "n": n, "n_recon": len(a), "n_native": len(b)}
    res = spearmanr(merged[f"{value_col}_recon"], merged[f"{value_col}_native"])
    return {
        "rho": float(res.statistic),  # type: ignore[reportAttributeAccessIssue]
        "p": float(res.pvalue),  # type: ignore[reportAttributeAccessIssue]
        "n": n,
        "n_recon": len(a),
        "n_native": len(b),
    }


def reachability_sweep(
    option_rows: pd.DataFrame, *, thresholds=(0.0, 0.5, 0.7, 0.85, 0.9, 0.95), min_options: int = 3
) -> dict:
    """The spec-3C reachability leg, generalized to a THRESHOLD GRID over the reconstructed SB360 option
    rows (built at reachability 0 -- i.e. ALL options). At each threshold, keep the chosen option plus the
    alternatives with ``completion >= threshold`` and recompute ``decision_pct`` / ``sel_efficiency``.

    'All visible teammates' over-counts unrealistic upfield options (``opponents_bypassed``-heavy, so
    EV-heavy via the progression term) that the keeper correctly declines, inverting ``decision_pct`` below
    0.5; a threshold matched to the provider's xPass distribution prunes them and restores responsiveness.
    Returns the responsiveness curve + the RECOMMENDED per-provider threshold (the lowest at which
    ``decision_pct`` crosses 0.5), alongside the shipped default (``reachability_min_xpass``, 0.85 -- the
    responsive value, since 0.5 inverts on SB360). A provider whose recommended threshold differs from the
    shipped default is a future ADR-009 apply; this driver only REPORTS.
    """
    from silly_kicks.gk_decision import GkDecisionParams, option_value

    params = GkDecisionParams()
    curve = []
    for thr in thresholds:
        beats, sels = [], []
        for _, g in option_rows.groupby(["game_id", "decision_id"], sort=False):
            is_chosen = g["is_chosen"].astype(bool).to_numpy()
            comp = pd.to_numeric(g["completion"], errors="coerce").to_numpy(dtype="float64")
            keep = g[is_chosen | (comp >= thr)]
            if int(keep["is_chosen"].astype(bool).sum()) != 1 or len(keep) < min_options:
                continue
            ev = option_value(keep, params=params).to_numpy(dtype="float64")
            ck = keep["is_chosen"].astype(bool).to_numpy()
            chosen = float(ev[ck][0])
            alt = ev[~ck]
            if not np.isfinite(chosen) or len(alt) == 0:
                continue
            beats.append((np.sum(alt < chosen) + 0.5 * np.sum(alt == chosen)) / len(alt))
            best = float(ev.max())
            sels.append(chosen / best if best > 0 else np.nan)
        curve.append(
            {
                "threshold": float(thr),
                "decision_pct_mean": float(np.mean(beats)) if beats else float("nan"),
                "decision_pct_t_vs_0.5": _one_sample_t(np.asarray(beats), 0.5) if len(beats) > 1 else float("nan"),
                "sel_efficiency_mean": float(np.nanmean(sels)) if sels else float("nan"),
                "n_decisions": len(beats),
            }
        )
    responsive = [c for c in curve if np.isfinite(c["decision_pct_mean"]) and c["decision_pct_mean"] > 0.5]
    return {
        "curve": curve,
        "recommended_threshold": (min(c["threshold"] for c in responsive) if responsive else None),
        "shipped_default": params.reachability_min_xpass,
    }


# --------------------------------------------------------------------------- owner-run orchestration
def _gk_ids_from_meta(meta: dict) -> list:
    return [p["id"] for p in meta.get("players", []) if (p.get("player_role") or {}).get("acronym") == "GK"]


def _download_first(dest, artifacts, tries, *, tok, base, mid):
    """First artifact path matching any (suffix, role) in ``tries``, downloaded if absent (or None)."""
    from scripts._loader_pining import _artifact_key, _download_to_temp

    for suffix, role in tries:
        try:
            key = _artifact_key(artifacts, suffix=suffix, role=role)
        except Exception:  # noqa: S112 -- this (suffix, role) is simply absent; try the next candidate
            continue
        if key not in artifacts:
            continue
        target = dest / str(artifacts[key])
        if not (target.exists() and target.stat().st_size > 0):
            _download_to_temp("skillcorner", mid, key, tok, base, dest).replace(target)
        return target
    return None


def _load_gi_matches(*, token, cache_dir, match_ids, max_matches):
    """Yield (match_id, gi_events, gk_ids) for owner-tier SkillCorner GI matches (both artifact schemas).

    Downloads the small GI artifacts (events csv/parquet + roster json) per match; raw data stays in
    the (gitignored) cache and is never committed. Aggregate outputs only are shareable.
    """
    from scripts._loader_pining import _base_url, _list_matches, _resolve_token

    tok = _resolve_token(token)
    base = _base_url()
    manifest = {m["id"]: m for m in _list_matches("skillcorner", tok, base)}
    ids = list(manifest) if match_ids is None else [m for m in match_ids if m in manifest]
    if max_matches is not None:
        ids = ids[:max_matches]
    root = Path(cache_dir) if cache_dir else Path("gk_decision_gi_cache")
    ev_tries = [("_dynamic_events.csv", "events"), (None, "events")]
    md_tries = [("_match.json", "metadata"), (None, "metadata")]
    for mid in ids:
        artifacts = manifest[mid]["artifacts"]
        dest = root / mid
        dest.mkdir(parents=True, exist_ok=True)
        ev_path = _download_first(dest, artifacts, ev_tries, tok=tok, base=base, mid=mid)
        md_path = _download_first(dest, artifacts, md_tries, tok=tok, base=base, mid=mid)
        if ev_path is None or md_path is None:
            continue
        gi = pd.read_parquet(ev_path) if ev_path.suffix == ".parquet" else pd.read_csv(ev_path, low_memory=False)
        meta = json.loads(md_path.read_text(encoding="utf-8"))
        yield mid, gi, _gk_ids_from_meta(meta)


def _measure_match(item) -> pd.DataFrame:
    """work(item): one match's native GK-decision samples (the for_each shard)."""
    from silly_kicks.gk_decision import SkillCornerGIOptionSet, compute_gk_decision_value
    from silly_kicks.providers.skillcorner import parse_passing_options

    mid, gi, gk_ids = item
    parsed = parse_passing_options(gi, game_id=mid)
    samples, _report = compute_gk_decision_value(SkillCornerGIOptionSet(parsed, keeper_ids=gk_ids))
    return samples


def _recon_samples(actions, frames, *, xpass, convention, reachability, visible_area=None, apply_bridge=False):
    """Reconstructed GK-decision samples for ONE match (owner-run; not CI-exercised).

    Derives the GK-distribution domain (``gk_distribution_mask``) and the keeper ids from the acting
    players of those actions, then scores via ``ReconstructedOptionSet``. ``apply_bridge`` stamps the
    real keeper id onto the anonymous SB360 actor row (ADR-078) BEFORE scoring.
    """
    from silly_kicks.gk_decision import GkDecisionParams, ReconstructedOptionSet, compute_gk_decision_value
    from silly_kicks.tracking import gk_distribution_mask

    if apply_bridge:
        from silly_kicks.keeper_identity import apply_actor_identities_to_frames

        frames = apply_actor_identities_to_frames(frames, actions)
    gk = actions[gk_distribution_mask(actions, frames, resolve_gk="robust").to_numpy()]
    keeper_ids = list(pd.unique(gk["player_id"].dropna()))  # the acting keepers (goalkick taker / acting GK)
    if not keeper_ids:
        return pd.DataFrame()
    os_ = ReconstructedOptionSet(
        gk,
        frames,
        xpass=xpass,
        params=GkDecisionParams(reachability_min_xpass=reachability),
        keeper_ids=keeper_ids,
        frame_convention=convention,
        visible_area=visible_area,
    )
    samples, _report = compute_gk_decision_value(os_)
    return samples


def _recon_option_rows(actions, frames, *, xpass, convention, visible_area=None, apply_bridge=False):
    """Raw reconstructed OPTION ROWS (reachability 0 -- ALL options) for ONE match, for the threshold
    sweep (which re-filters + recomputes decision_pct at each grid point). Same domain/keeper derivation
    + actor bridge as ``_recon_samples``.
    """
    from silly_kicks.gk_decision import GkDecisionParams, ReconstructedOptionSet
    from silly_kicks.tracking import gk_distribution_mask

    if apply_bridge:
        from silly_kicks.keeper_identity import apply_actor_identities_to_frames

        frames = apply_actor_identities_to_frames(frames, actions)
    gk = actions[gk_distribution_mask(actions, frames, resolve_gk="robust").to_numpy()]
    keeper_ids = list(pd.unique(gk["player_id"].dropna()))
    if not keeper_ids:
        return pd.DataFrame()
    os_ = ReconstructedOptionSet(
        gk,
        frames,
        xpass=xpass,
        params=GkDecisionParams(reachability_min_xpass=0.0),
        keeper_ids=keeper_ids,
        frame_convention=convention,
        visible_area=visible_area,
    )
    return os_.option_rows()


def _per_keeper_match(samples: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Per-(keeper, game_id) mean of ``value_col`` -- the join grain for the Rosetta-Stone fidelity."""
    if samples.empty:
        return pd.DataFrame(columns=["keeper", "game_id", value_col])
    return samples.groupby(["keeper", "game_id"], dropna=False)[value_col].mean().reset_index()


def _reconstruction_verdicts(*, token, cache_dir, sc_ids, sb_ids, max_matches) -> dict:
    """Owner-run reconstruction legs (NOT CI-exercised -- needs the owner-tier tracking + SB360 corpora).

    Fidelity: per SkillCorner match, native-GI samples + reconstructed-from-tracking samples through the
    SAME engine, aggregated per (keeper, game_id), Spearman (the RANKING what chosen-vs-available needs;
    native GI and reconstructed keep different per-decision ids, so the Rosetta-Stone join is per keeper).
    SB360: naive (reachability 0) vs reachability-filtered reconstructed option sets -> the spec-3C sweep.
    """
    from scripts._loader_pining import load_matches, load_statsbomb_matches
    from silly_kicks.expected_passing import PassCompletionModel
    from silly_kicks.gk_decision import GkDecisionParams

    xpass = PassCompletionModel.bundled()
    default_reach = GkDecisionParams().reachability_min_xpass  # fidelity measured at the SHIPPED default (0.85)
    out: dict = {"fidelity": {}, "sb360_reachability_sweep": {}}

    # -- Rosetta-Stone fidelity (SkillCorner native GI + tracking) --
    native_rows, recon_rows = [], []
    gi_by_id = {
        mid: (gi, gk)
        for mid, gi, gk in _load_gi_matches(token=token, cache_dir=cache_dir, match_ids=sc_ids, max_matches=max_matches)
    }
    for _prov, mid, actions, frames, _home in load_matches(
        providers=["skillcorner"],
        match_ids={"skillcorner": sc_ids} if sc_ids else None,
        token=token,
        max_per_provider=max_matches,
        cache_dir=cache_dir,
    ):
        if mid not in gi_by_id:
            continue
        gi, gk_ids = gi_by_id[mid]
        native_rows.append(_measure_match((mid, gi, gk_ids)))
        recon_rows.append(
            _recon_samples(actions, frames, xpass=xpass, convention="match_ltr", reachability=default_reach)
        )
    native = pd.concat([r for r in native_rows if not r.empty], ignore_index=True) if native_rows else pd.DataFrame()
    recon = pd.concat([r for r in recon_rows if not r.empty], ignore_index=True) if recon_rows else pd.DataFrame()
    if not native.empty and not recon.empty:
        for c in ("decision_value", "sel_efficiency"):
            out["fidelity"][c] = fidelity_spearman(
                _per_keeper_match(recon, c), _per_keeper_match(native, c), value_col=c, keys=["keeper", "game_id"]
            )

    # -- SB360 reachability THRESHOLD sweep (grid; recommends the per-provider threshold, ADR-009) --
    sb_option_rows = []
    for _prov, _mid, actions, frames, _home, visible_area in load_statsbomb_matches(
        match_ids=sb_ids, token=token, max_matches=max_matches, cache_dir=cache_dir
    ):
        rows = _recon_option_rows(
            actions, frames, xpass=xpass, convention="per_action_ltr", visible_area=visible_area, apply_bridge=True
        )
        if not rows.empty:
            sb_option_rows.append(rows)
    sb_rows = pd.concat(sb_option_rows, ignore_index=True) if sb_option_rows else pd.DataFrame()
    if not sb_rows.empty:
        out["sb360_reachability_sweep"] = reachability_sweep(sb_rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="output dir (not needed with --list-matches)")
    ap.add_argument("--token", default=None, help="pining token (else resolved from the environment)")
    ap.add_argument("--cache-dir", default=None, help="persistent GI-artifact cache (gitignored)")
    ap.add_argument("--match-ids-json", default=None, help='JSON ["<id>", ...] pinning WHICH matches (parallel split)')
    ap.add_argument("--max-matches", type=int, default=None)
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (dev only; artifact is marked)")
    ap.add_argument("--list-matches", action="store_true", help="print available match ids as JSON and exit")
    ap.add_argument(
        "--reconstruction",
        action="store_true",
        help="ALSO run the owner-run reconstruction legs (SkillCorner fidelity + SB360 sweep)",
    )
    ap.add_argument("--sb-match-ids-json", default=None, help='JSON ["<id>", ...] pinning SB360 matches for the sweep')
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

    match_ids = json.loads(Path(args.match_ids_json).read_text(encoding="utf-8")) if args.match_ids_json else None

    if args.list_matches:
        ids = [
            mid
            for mid, _gi, _gk in _load_gi_matches(
                token=args.token, cache_dir=args.cache_dir, match_ids=match_ids, max_matches=args.max_matches
            )
        ]
        print(json.dumps(ids, indent=2))
        return

    from scripts._driver import for_each

    dest = Path(args.out)

    def _matches():
        yield from _load_gi_matches(
            token=args.token, cache_dir=args.cache_dir, match_ids=match_ids, max_matches=args.max_matches
        )

    res = for_each(
        _matches(),
        key=lambda item: ("skillcorner", str(item[0])),
        work=_measure_match,
        shard_root=dest / "shards",
        token_inputs={"metric": "gk_decision_native", "schema": "gk-decision-native-1"},
        label="match",
    )
    shard_files = sorted(res.shard_dir.glob("*.parquet"))
    samples = pd.concat([pd.read_parquet(s) for s in shard_files], ignore_index=True) if shard_files else pd.DataFrame()
    verdicts: dict = reduce_samples(samples) if not samples.empty else {"n_decisions": 0}

    if args.reconstruction:
        sb_ids = (
            json.loads(Path(args.sb_match_ids_json).read_text(encoding="utf-8")) if args.sb_match_ids_json else None
        )
        verdicts["reconstruction"] = _reconstruction_verdicts(
            token=args.token, cache_dir=args.cache_dir, sc_ids=match_ids, sb_ids=sb_ids, max_matches=args.max_matches
        )

    contract = declare_inputs(
        driver="validate_gk_decision",
        metric_columns=list(_DECISION_METRICS),
        option_value="completion_progression",
        min_per_keeper=_MIN_PER_KEEPER,
        reconstruction_seams=[
            "PassCompletionModel",
            "compute_packing_metrics",
            "action_ltr_goal_map",
            "reachability_min_xpass",
        ],
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
