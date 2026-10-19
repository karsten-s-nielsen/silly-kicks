"""Construct-validity battery for the TF-46 Arm 1 corner-offence metrics (reported, not gated).

Over the owner corpus: per-corner routine features (fitted CornerRoutineModel) + per-routine chance
quality, then the three-gate reading (responsiveness / discrimination-vs-permutation-null net of team /
transfer) plus the SB360 initial-snapshot divergence (section 12.7) and the convergent-timing agreement
(section 12.6). Descriptive, no accuracy gate (the encoder/NMF are unsupervised; name-by-mechanism).

Legs (each pre-registered, aggregate-only; reversibility policy -- no reversible ids committed):
- Responsiveness: per-metric corpus mean of the per-routine chance quality (moves off baseline).
- Discrimination: one-way ICC by provider-qualified team vs a label-permutation null.
- Net of team: team one-way + opponent-strength leave-one-out residual + routine fixed-effect residual
  (does the team signal survive netting out the defence faced and the routine mix).
- Transfer: season-to-season sign agreement + residual correlation (does a team's corner chance quality
  in one season predict the next, net of that season's league baseline).
- Section 12.5 cross-provider stability: per-provider means of each metric.
- Section 12.6 convergent timing: TF-79 first-contact vs the SkillCorner GI first player-possession time.
- Section 12.7 SB360 init divergence: the -2s-vs-kick-frame positional delta + initial-zone reassignment
  on IDSSE (a provider carrying continuous tracking at BOTH the -2s and kick-frame snapshots).

A power check takes an InjectionSpec recipe, never an observed-outcome vector, so the observed outcome is
unrepresentable. Clean-tree guard first (ADR-037); input contract declares the seams (ADR-056). Real
corpus run is opt-in / e2e (DGX or media-PC); tests/scripts cover the pure kernels. Memo lands at
docs/research/corner_routines/ (README.md + metrics.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

_MIN_PER_TEAM = 5
_RNG_SEED = 20261016


def icc_vs_permutation(
    df: pd.DataFrame, value_col: str, group_col: str, *, n_perm: int = 1000, seed: int = _RNG_SEED
) -> dict:
    """One-way ICC of ``value_col`` by ``group_col`` vs a label-permutation null (reported, not gated).

    ICC = between-group variance share. The null permutes group labels ``n_perm`` times; ``null_p95`` is
    its 95th percentile and ``p`` the fraction of null ICCs >= the observed.
    """
    sub = df[[value_col, group_col]].dropna()
    if sub[group_col].nunique() < 2 or len(sub) < 2:
        return {"icc": float("nan"), "null_p95": float("nan"), "p": float("nan"), "n": len(sub)}

    def _icc(vals: np.ndarray, groups: np.ndarray) -> float:
        grand = vals.mean()
        total = float(((vals - grand) ** 2).sum())
        if total <= 0:
            return 0.0
        between = 0.0
        for g in np.unique(groups):
            v = vals[groups == g]
            between += len(v) * (v.mean() - grand) ** 2
        return float(between / total)

    vals = sub[value_col].to_numpy(dtype=float)
    groups = sub[group_col].to_numpy()
    obs = _icc(vals, groups)
    rng = np.random.default_rng(seed)
    null = np.array([_icc(vals, rng.permutation(groups)) for _ in range(n_perm)])
    return {
        "icc": obs,
        "null_p95": float(np.percentile(null, 95)),
        "p": float((null >= obs).mean()),
        "n": len(sub),
    }


def opponent_adjusted(df: pd.DataFrame, value_col: str, atk_col: str, def_col: str) -> pd.DataFrame:
    """Net-of-opponent residual: value minus the LEAVE-ONE-ATTACKER-OUT defending-team mean.

    The club-adjusted analog for a team-subject metric (``validate_gk_decision.club_adjusted_residuals``
    with keeper->attacking-team, team->defending-team): a corner's value minus how much chance quality the
    defending team concedes to OTHER attacking teams. Removes "faced weak defences". Returns the dropna'd
    frame with a ``_resid`` column (NaN where the defending team has no other-attacker baseline).
    """
    s = df.dropna(subset=[value_col, def_col]).copy()
    s["_v"] = s[value_col].astype(float)
    dsum = s.groupby(def_col, observed=True)["_v"].transform("sum")
    dcnt = s.groupby(def_col, observed=True)["_v"].transform("count")
    csum = s.groupby([def_col, atk_col], observed=True)["_v"].transform("sum")
    ccnt = s.groupby([def_col, atk_col], observed=True)["_v"].transform("count")
    denom = dcnt - ccnt
    loo = (dsum - csum) / denom.where(denom > 0)
    s["_resid"] = s["_v"] - loo
    return s


def team_season_transfer(
    df: pd.DataFrame, value_col: str, team_col: str, season_col: str, *, min_per: int = _MIN_PER_TEAM
) -> dict:
    """Season-to-season transfer: a team's season residual (team-season mean minus the leave-the-team-out
    SEASON-league baseline) in its two most-sampled seasons, correlated across teams (>50% / positive =>
    the corner chance-quality signal travels with the team across seasons, net of the league baseline)."""
    rows = []
    s = df.dropna(subset=[value_col, season_col])
    for season, sg in s.groupby(season_col, observed=True):
        ssum, scnt = float(sg[value_col].sum()), len(sg)
        for team, tg in sg.groupby(team_col, observed=True):
            if len(tg) < min_per or (scnt - len(tg)) <= 0:
                continue
            loo = (ssum - float(tg[value_col].sum())) / (scnt - len(tg))
            rows.append({"team": team, "season": season, "n": len(tg), "resid": float(tg[value_col].mean()) - loo})
    kt = pd.DataFrame(rows)
    if kt.empty:
        return {"n_transfer_teams": 0, "sign_agreement": float("nan"), "resid_corr": float("nan")}
    multi = kt.groupby("team").filter(lambda g: g["season"].nunique() >= 2)
    pairs = []
    for _team, g in multi.groupby("team"):
        g2 = g.sort_values("n", ascending=False).head(2)
        if len(g2) == 2:
            pairs.append((float(g2["resid"].iloc[0]), float(g2["resid"].iloc[1])))
    if len(pairs) < 3:
        return {"n_transfer_teams": len(pairs), "sign_agreement": float("nan"), "resid_corr": float("nan")}
    ra = np.array([p[0] for p in pairs])
    rb = np.array([p[1] for p in pairs])
    from scipy.stats import spearmanr

    res = spearmanr(ra, rb)
    return {
        "n_transfer_teams": len(pairs),
        "sign_agreement": float(np.mean(np.sign(ra) == np.sign(rb))),
        "resid_corr": float(res.statistic),  # type: ignore[reportAttributeAccessIssue]
    }


def convergent_timing(df: pd.DataFrame, contact_col: str, possession_col: str) -> dict:
    """Section 12.6: agreement between TF-79 first-contact time and the GI first player-possession time.

    Two independent onset measures of the same latent (when the delivery is first met); Spearman rho +
    the signed offset distribution. Reported, not gated. NaN with <3 paired corners.
    """
    s = df[[contact_col, possession_col]].dropna()
    n = len(s)
    if n < 3:
        nan = float("nan")
        return {"n": n, "rho": nan, "p": nan, "offset_mean_s": nan, "offset_median_s": nan}
    from scipy.stats import spearmanr

    res = spearmanr(s[contact_col], s[possession_col])
    off = (s[possession_col] - s[contact_col]).to_numpy(dtype=float)
    return {
        "n": n,
        "rho": float(res.statistic),  # type: ignore[reportAttributeAccessIssue]
        "p": float(res.pvalue),  # type: ignore[reportAttributeAccessIssue]
        "offset_mean_s": float(off.mean()),
        "offset_median_s": float(np.median(off)),
    }


def sb360_init_divergence(init_xy: np.ndarray, kick_xy: np.ndarray, model) -> dict:
    """Section 12.7: bound the SB360 train/serve mismatch. ``init_xy`` is the tracking -2s snapshot and
    ``kick_xy`` the kick-frame snapshot for the SAME players; report the positional-delta distribution
    and the initial-zone reassignment rate (argmax initial zone differs between the two snapshots)."""
    from silly_kicks.corners._zones import _gaussian_responsibilities

    init_xy = np.asarray(init_xy, dtype=float).reshape(-1, 2)
    kick_xy = np.asarray(kick_xy, dtype=float).reshape(-1, 2)
    delta = np.hypot(kick_xy[:, 0] - init_xy[:, 0], kick_xy[:, 1] - init_xy[:, 1])
    r_init = _gaussian_responsibilities(init_xy, model.init_means, model.init_covs, model.init_weights)
    r_kick = _gaussian_responsibilities(kick_xy, model.init_means, model.init_covs, model.init_weights)
    reassigned = (r_init.argmax(axis=1) != r_kick.argmax(axis=1)).mean() if len(init_xy) else float("nan")
    return {
        "n_players": len(init_xy),
        "delta_mean_m": float(delta.mean()) if len(delta) else float("nan"),
        "delta_p95_m": float(np.percentile(delta, 95)) if len(delta) else float("nan"),
        "zone_reassignment_rate": float(reassigned),
    }


def dominant_routine(coeffs: np.ndarray) -> np.ndarray:
    """Per-corner argmax NMF routine (the routine-fixed-effect bucket). An all-NaN (short/unclear) row
    keeps its own -1 bucket (ADR-042); ``np.nanargmax`` raises on an all-NaN slice, so it runs only on the
    rows that carry a signal."""
    coeffs = np.asarray(coeffs, dtype=float)
    dom = np.full(coeffs.shape[0], -1, dtype=np.int64)
    has_signal = ~np.isnan(coeffs).all(axis=1)
    if has_signal.any():
        dom[has_signal] = np.nanargmax(coeffs[has_signal], axis=1)
    return dom


def derive_season(date: object) -> str | None:
    """European season label ("YYYY/YYYY") from a match date, split at July (month >= 7 -> that year/+1)."""
    if date is None or (isinstance(date, float) and np.isnan(date)):
        return None
    ds = str(date)
    if len(ds) < 7 or not ds[:4].isdigit() or not ds[5:7].isdigit():
        return None
    y, m = int(ds[:4]), int(ds[5:7])
    return f"{y}/{y + 1}" if m >= 7 else f"{y - 1}/{y}"


#: Chance-quality currencies, most-robust first. n_goals / n_shots are the MODEL-FREE backbone
#: (fully observed, provider-uniform, no xG model -- silly-kicks ships none by design, ADR-110);
#: sum_xg is a provider-native-xG companion present only where the provider supplies xG (statsbomb
#: statsbomb_xg, sportec shot_x_g) and is read PER PROVIDER, never pooled across providers' xG models.
_CHANCE_METRICS = ("n_goals", "n_shots", "sum_xg")


def reduce_samples(samples: pd.DataFrame) -> dict:
    """Pooled verdicts over all corpus corner-feature/chance-quality rows (the REDUCE).

    net_of_team / transfer legs are computed only when their confound columns are present
    (``defending_team_id`` / ``dom_routine`` / ``season``), so a minimal sample frame still reduces.
    """
    from silly_kicks.id_compat import canonical_id_series

    samples = samples.copy()
    # Team ids are PER-PROVIDER namespaces AND mixed dtype across providers (SkillCorner str,
    # statsbomb/idsse int). A bare pooled team_id both mis-merges different-provider teams that share a
    # spelling and crashes np.unique's sort (int vs str). Group discrimination on a provider-qualified
    # CANONICAL key (ADR-019): canonical_id normalizes the dtype, the provider prefix keeps the namespace.
    have_team = "team_id" in samples.columns
    prov_prefix = samples["provider"].astype("string") + ":" if "provider" in samples.columns else ""
    if have_team:
        samples["_team_key"] = prov_prefix + canonical_id_series(samples["team_id"]).astype("string")
    if "defending_team_id" in samples.columns:
        samples["_def_key"] = prov_prefix + canonical_id_series(samples["defending_team_id"]).astype("string")
    out: dict = {"n_corners": len(samples), "n_teams": int(samples["_team_key"].nunique()) if have_team else 0}
    metrics = [m for m in _CHANCE_METRICS if m in samples.columns]

    def _mean(frame: pd.DataFrame, col: str) -> float:
        vals = frame[col].to_numpy(dtype=float)
        # all-NaN is legitimate (e.g. SkillCorner sum_xg: no native shot xG) -> honest NaN, no warning
        return float(np.nanmean(vals)) if len(vals) and not bool(np.isnan(vals).all()) else float("nan")

    # Responsiveness: per-metric corpus mean (the per-routine quality moves off baseline in the memo).
    out["responsiveness"] = {f"{m}_mean": _mean(samples, m) for m in metrics}
    # Discrimination: one-way ICC of each metric by provider-qualified team vs a label-permutation null.
    if have_team:
        out["discrimination_one_way"] = {m: icc_vs_permutation(samples, m, "_team_key") for m in metrics}
    # Net of team: team one-way + opponent leave-one-out residual + routine fixed-effect residual.
    if have_team and "_def_key" in samples.columns:
        out["net_of_team"] = {}
        for m in metrics:
            legs = {"team_one_way": icc_vs_permutation(samples, m, "_team_key")}
            opp = opponent_adjusted(samples, m, "_team_key", "_def_key")
            legs["opponent_adjusted"] = icc_vs_permutation(opp.dropna(subset=["_resid"]), "_resid", "_team_key")
            if "dom_routine" in samples.columns:
                rfe = samples.dropna(subset=[m]).copy()
                rfe["_resid"] = rfe[m].astype(float) - rfe.groupby("dom_routine", observed=True)[m].transform("mean")
                legs["routine_fixed_effect"] = icc_vs_permutation(rfe.dropna(subset=["_resid"]), "_resid", "_team_key")
            out["net_of_team"][m] = legs
    # Transfer: season-to-season sign agreement + residual correlation (net of the season-league baseline).
    if have_team and "season" in samples.columns:
        out["transfer"] = {m: team_season_transfer(samples, m, "_team_key", "season") for m in metrics}
    # Section 12.5 cross-provider stability: per-provider means of each metric (SkillCorner broadcast vs
    # optical full-obs vs SB360 freeze). Reported, not gated. sum_xg is per-provider by construction.
    if "provider" in samples.columns:
        out["cross_provider"] = {
            str(p): {"n": len(g), **{f"{m}_mean": _mean(g, m) for m in metrics}}
            for p, g in samples.groupby("provider", observed=True)
        }
    # Section 12.6 convergent timing: TF-79 first-contact vs GI first player-possession (SkillCorner only).
    if {"first_contact_time", "first_possession_time"}.issubset(samples.columns):
        out["convergent_timing"] = convergent_timing(samples, "first_contact_time", "first_possession_time")
    # TF46-IMPL-06: attribute a dark GI timing leg -- matches whose GI events fetch/parse FAILED (vs the
    # honest absence of a GI artifact, which stays NaN and is NOT counted here).
    if "gi_fetch_failed" in samples.columns and "game_id" in samples.columns:
        out["n_matches_gi_fetch_failed"] = int(
            samples.loc[samples["gi_fetch_failed"].astype(bool), "game_id"].nunique()
        )
    return out


def _work(item) -> pd.DataFrame:
    """Per-match per-corner routine-feature + chance-quality rows (requires fitted models in --models)."""
    # Corpus orchestration is exercised at Task 14 (DGX); kept thin so the pure kernels above stay the
    # CI-tested surface. The fitted models are loaded in main() and threaded via a closure there.
    raise NotImplementedError("validate_corners._work is wired by main(); run the e2e corpus pass on the DGX.")


def gi_possession_times(gi: pd.DataFrame) -> pd.DataFrame:
    """Per SkillCorner GI player-possession event: ``(period, t_rel)`` where ``t_rel`` is seconds RELATIVE
    TO THE PERIOD START.

    GI carries ``minute_start``/``second_start`` on a MATCH-ABSOLUTE clock (period 2 continues past 45:00,
    e.g. 2701-5810 s), while the SPADL corner ``t_kick`` is PERIOD-RELATIVE (ADR-017). Rebase GI by
    subtracting each period's earliest possession time (its kickoff proxy, ~0-2 s in) so the two clocks
    compare. ``time_start`` is a "MM:SS.s" STRING and is deliberately not used.
    """
    if gi is None or gi.empty or "event_type" not in gi.columns:
        return pd.DataFrame(columns=["period", "t_rel"])
    pp = gi[gi["event_type"] == "player_possession"]
    if pp.empty or not {"minute_start", "second_start", "period"}.issubset(pp.columns):
        return pd.DataFrame(columns=["period", "t_rel"])
    t_abs = pp["minute_start"].astype(float) * 60.0 + pp["second_start"].astype(float)
    out = pd.DataFrame({"period": pp["period"].astype(int).to_numpy(), "t_abs": t_abs.to_numpy()})
    out["t_rel"] = out["t_abs"] - out.groupby("period")["t_abs"].transform("min")
    return out[["period", "t_rel"]]


def gi_first_possession(pt: pd.DataFrame, period: int, t_kick: float) -> float:
    """First GI player-possession (period-relative) time at/after ``t_kick`` in ``period`` (NaN if none)."""
    if pt.empty:
        return float("nan")
    hit = pt[(pt["period"] == period) & (pt["t_rel"] >= t_kick)]
    return float(hit["t_rel"].min()) if not hit.empty else float("nan")


def _divergence_rows(item, zone_model) -> pd.DataFrame:
    """Section 12.7: per-(corner, attacking player) reflected -2s + kick-frame positions for one IDSSE match.

    Mirrors ``build_corner_routines._snapshot_rows`` but the SECOND snapshot is the kick frame (t_kick),
    not the target, so the pooled reduce can bound the -2s-vs-kick-frame (SB360 serve) mismatch.
    """
    from silly_kicks.corners import add_corner_windows, identify_corners
    from silly_kicks.corners._encode import _nearest_frame_xy, _reflect_y
    from silly_kicks.tracking._ball_contact import detect_ball_contacts

    cols = ["init_x", "init_y", "kick_x", "kick_y"]
    if zone_model is None or item.frames is None or item.frames.empty:
        return pd.DataFrame(columns=cols)
    corners = identify_corners(item.actions, provider=item.provider)
    corners = corners[corners["corner_kind"] == "direct"]
    if corners.empty:
        return pd.DataFrame(columns=cols)
    windowed = add_corner_windows(corners, item.frames, detect_ball_contacts(item.frames))
    rows: list[dict] = []
    for _, c in windowed.iterrows():
        period, team, side = int(c["period_id"]), c["attacking_team_id"], str(c["corner_side"])
        init = _nearest_frame_xy(item.frames, period, float(c["t_init"]), team)
        kick = _nearest_frame_xy(item.frames, period, float(c["t_kick"]), team)
        if init is None or kick is None:
            continue
        init, kick = _reflect_y(init, side), _reflect_y(kick, side)
        merged = init.merge(kick, on="player_id", suffixes=("_i", "_k"))
        for _, p in merged.iterrows():
            rows.append(
                {
                    "init_x": float(p["x_i"]),
                    "init_y": float(p["y_i"]),
                    "kick_x": float(p["x_k"]),
                    "kick_y": float(p["y_k"]),
                }
            )
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=None, help="dir holding the fitted run_zone/ + routine/ bundles")
    ap.add_argument("--token", default=None)
    ap.add_argument("--cache-dir", default=None)
    # "sportec" is the same data as "idsse" (served under that pining key) and 404s the listing.
    ap.add_argument("--providers", nargs="*", default=["skillcorner", "idsse", "statsbomb"])
    ap.add_argument("--match-ids-json", default=None, help='per-provider pins: JSON {"<provider>": ["<id>", ...]}')
    ap.add_argument("--max-matches", type=int, default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    from scripts._driver import for_each, shard_path
    from scripts._input_contract import declare_inputs
    from scripts._loader_pining import _base_url, _list_matches, _resolve_token, pining_source
    from scripts._provenance import git_provenance, require_clean_tree
    from scripts.validate_gk_decision import _GI_EV_TRIES, _download_first
    from silly_kicks.corners import (
        CornerRoutineModel,
        RunZoneModel,
        add_corner_windows,
        compute_corner_chance_quality,
        compute_corner_routine_features,
        compute_corner_run_encoding,
        identify_corners,
    )
    from silly_kicks.id_compat import ids_differ
    from silly_kicks.tracking._ball_contact import detect_ball_contacts

    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)
    match_ids = json.loads(Path(args.match_ids_json).read_text(encoding="utf-8")) if args.match_ids_json else None
    models_dir = Path(args.models) if args.models else None
    zone_model = RunZoneModel.load(models_dir / "run_zone") if models_dir else None
    routine_model = CornerRoutineModel.load(models_dir / "routine") if models_dir else None
    dest = Path(args.out)

    # Manifest-derived season (date -> European season, for the transfer leg) + SkillCorner GI artifact
    # keys (for the section 12.6 convergent-timing leg). Listed ONCE here; no LoadedMatch change.
    tok = _resolve_token(args.token)
    base = _base_url()
    season_map: dict[str, str | None] = {}
    gi_artifacts: dict[str, dict] = {}
    n_manifest_listing_failed = 0  # TF46-IMPL-06: a swallowed listing failure is COUNTED, not silent
    for provider in args.providers:
        try:
            manifest = list(_list_matches(provider, tok, base))
        except Exception:
            n_manifest_listing_failed += 1
            continue
        for m in manifest:
            mid = str(m["id"])
            season_map[mid] = derive_season(m.get("date"))
            if provider == "skillcorner":
                gi_artifacts[mid] = dict(m.get("artifacts", {}))
    gi_cache = (Path(args.cache_dir) if args.cache_dir else dest) / "gi_events"

    def _match_rows(item) -> pd.DataFrame:
        if zone_model is None or routine_model is None or item.frames is None or item.frames.empty:
            return pd.DataFrame()
        corners = identify_corners(item.actions, provider=item.provider)
        windowed = add_corner_windows(corners, item.frames, detect_ball_contacts(item.frames))
        encoding = compute_corner_run_encoding(windowed, item.frames, model=zone_model)
        feats = compute_corner_routine_features(encoding, model=routine_model)
        # Dominant routine (argmax of the NMF coefficients) -> the routine-fixed-effect bucket; a NaN
        # (short/unclear) corner keeps its own -1 bucket (ADR-042: never dropped).
        feat_cols = [c for c in feats.columns if c.startswith("routine_feat_")]
        feats["dom_routine"] = dominant_routine(feats[feat_cols].to_numpy(dtype=float))
        # n_shots/n_goals are model-free (always computed); sum_xg only where the provider injects xG.
        xg_col = "xg" if "xg" in item.actions.columns else None
        quality = compute_corner_chance_quality(windowed, item.actions, xg_column=xg_col)
        keep = ["game_id", "corner_id", "attacking_team_id", "period_id", "t_kick", "first_contact_time"]
        rows = feats.merge(windowed[keep], on=["game_id", "corner_id"])
        rows = rows.rename(columns={"attacking_team_id": "team_id"})
        rows["provider"] = item.provider  # for the section 12.5 cross-provider stability leg
        # Defending team = the match's OTHER team (two team ids in the frames); net-of-opponent leg.
        match_teams = list(pd.unique(item.frames["team_id"].dropna()))
        if len(match_teams) == 2:
            a, b = match_teams
            # Defending team = the match's OTHER team. ids_differ is NA-safe + dtype-safe (ADR-019) and
            # Series-valued: attacking == a -> defending b, else a.
            is_a = ~ids_differ(rows["team_id"], pd.Series([a] * len(rows), index=rows.index)).to_numpy()
            rows["defending_team_id"] = np.where(is_a, b, a)
        else:
            rows["defending_team_id"] = pd.NA
        # Section 12.6: first GI player-possession time per corner (SkillCorner GI only), joined on timing.
        # TF46-IMPL-06: gi_fetch_failed distinguishes a genuine FETCH/PARSE failure (counted) from honest
        # data-absence (no GI events artifact) so a dark timing leg is attributable, not silently NaN.
        rows["first_possession_time"] = np.nan
        rows["gi_fetch_failed"] = False
        if item.provider == "skillcorner" and str(item.actions["game_id"].iloc[0]) in gi_artifacts:
            gid = str(item.actions["game_id"].iloc[0])
            try:
                gi_dir = gi_cache / gid
                gi_dir.mkdir(parents=True, exist_ok=True)  # _download_first writes here but does not create it
                ev = _download_first(gi_dir, gi_artifacts[gid], _GI_EV_TRIES, tok=tok, base=base, mid=gid)
                gi = None
                if ev is not None:
                    gi = pd.read_parquet(ev) if str(ev).endswith(".parquet") else pd.read_csv(ev, low_memory=False)
                if gi is not None:
                    pt = gi_possession_times(gi)
                    rows["first_possession_time"] = [
                        gi_first_possession(pt, int(pid), float(tk))
                        for pid, tk in zip(rows["period_id"], rows["t_kick"], strict=True)
                    ]
                # ev is None == the match carries no GI events artifact: honest absence, NOT a failure.
            except Exception:
                rows["gi_fetch_failed"] = True
        return rows.merge(quality, on=["game_id", "corner_id"], how="left")

    refs, load = pining_source(
        providers=args.providers,
        match_ids=match_ids,
        max_per_provider=args.max_matches,
        cache_dir=args.cache_dir,
        token=args.token,
        with_velocity=False,  # corners read x/y only; skip the ~32s/match velocity derivation
    )
    res = for_each(
        refs,
        key=lambda r: r.key,
        load=load,
        work=_match_rows,
        shard_root=dest / "shards",
        token_inputs={"metric": "corner_validity", "schema": "corner-validity-3"},
        label="match",
    )
    parts = [f for f in (pd.read_parquet(shard_path(res.shard_dir, k)) for k in res.shard_keys) if not f.empty]
    samples = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not samples.empty:
        samples["season"] = samples["game_id"].map(lambda g: season_map.get(str(g)))
    verdicts: dict = reduce_samples(samples) if not samples.empty else {"n_corners": 0}
    verdicts["n_manifest_listing_failed"] = n_manifest_listing_failed  # TF46-IMPL-06: provider listings that errored

    # Section 12.7 SB360 init divergence -- a SEPARATE pass over the IDSSE refs only (continuous tracking
    # carries both the -2s and kick-frame snapshots). Pool the per-player deltas, score against the fitted
    # init zones. Skipped when IDSSE is not in --providers or no models are given.
    if zone_model is not None and "idsse" in args.providers:
        d_refs, d_load = pining_source(
            providers=["idsse"],
            match_ids={"idsse": match_ids["idsse"]} if match_ids and "idsse" in match_ids else None,
            max_per_provider=args.max_matches,
            cache_dir=args.cache_dir,
            token=args.token,
            with_velocity=False,
        )
        d_res = for_each(
            d_refs,
            key=lambda r: r.key,
            load=d_load,
            work=lambda m: _divergence_rows(m, zone_model),
            shard_root=dest / "divergence_shards",
            token_inputs={"metric": "corner_init_divergence", "schema": "corner-divergence-1"},
            label="match",
        )
        d_cols = ["init_x", "init_y", "kick_x", "kick_y"]
        d_parts = [
            f for f in (pd.read_parquet(shard_path(d_res.shard_dir, k)) for k in d_res.shard_keys) if not f.empty
        ]
        div = pd.concat(d_parts, ignore_index=True) if d_parts else pd.DataFrame(columns=d_cols)
        if not div.empty:
            verdicts["sb360_init_divergence"] = sb360_init_divergence(
                div[["init_x", "init_y"]].to_numpy(dtype=float),
                div[["kick_x", "kick_y"]].to_numpy(dtype=float),
                zone_model,
            )

    contract = declare_inputs(
        driver="validate_corners",
        min_per_team=_MIN_PER_TEAM,
        seams=["RunZoneModel", "CornerRoutineModel", "compute_corner_chance_quality"],
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
