"""``compute_team_kpis`` -- the TF-52 orchestrator.

Assembles the pressing / progression / build-up families over a single possession foundation, emits
one row per ``(game_id, team_id)``, adds the within-Ns post-recovery companions, and conserves the
match + shot-xG population in a :class:`TeamKpiReport`. Pure: never mutates ``actions``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.id_compat import canonical_id, canonical_id_series, ids_match
from silly_kicks.spadl import config as spadlconfig

from ._buildup import compute_buildup_kpis
from ._columns import (
    POST_RECOVERY_SOURCE_COLUMNS,
    TEAM_KPI_COLUMNS,
    TEAM_KPI_KEYS,
)
from ._config import TeamKpiParams
from ._possession import add_possession_context, build_spells, possession_minutes
from ._pressing import compute_pressing_kpis
from ._progression import compute_progression_kpis
from ._report import TeamKpiReport

_FL = spadlconfig.field_length
_FW = spadlconfig.field_width
_SHOT_TYPE_IDS = [spadlconfig.actiontype_id[n] for n in ("shot", "shot_penalty", "shot_freekick")]
_NON_PENALTY_SHOT_TYPE_IDS = [spadlconfig.actiontype_id[n] for n in ("shot", "shot_freekick")]
_BOX_MIN_X = _FL - spadlconfig.penalty_area_depth
_BOX_HALF_W = spadlconfig.penalty_area_half_width
_FT_MIN_X = 2.0 / 3.0 * _FL
_DEFAULT_PARAMS = TeamKpiParams.default()  # module-level singleton (avoids a call in the arg default)


def compute_team_kpis(
    actions: pd.DataFrame,
    *,
    xg_column: str | None = None,
    params: TeamKpiParams = _DEFAULT_PARAMS,
) -> tuple[pd.DataFrame, TeamKpiReport]:
    """Per-``(game_id, team_id)`` event-only team KPIs + a conserving census.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions (``type_id`` / ``result_id`` int-coded) for one or more matches.
    xg_column : str | None
        Name of an injected pre-shot xG column; only the high-opportunity-shots KPI reads it.
    params : TeamKpiParams
        Frozen thresholds (defaults from :meth:`TeamKpiParams.default`).

    Returns
    -------
    (samples, report)
        ``samples``: one row per ``(game_id, team_id)`` over the columns in ``TEAM_KPI_COLUMNS``.
        ``report``: a :class:`TeamKpiReport` conserving the match + shot-xG population.

    Examples
    --------
    Compute team KPIs for a match of SPADL ``actions`` (int-coded ``type_id`` / ``result_id``), with
    an injected pre-shot xG column::

        samples, report = compute_team_kpis(actions, xg_column="xg")
        home_ppda = samples.loc[samples["team_id"] == home_team_id, "ppda"]
        assert report.n_matches_scored + report.n_matches_excluded_not_two_teams == report.n_matches_in
    """
    games = list(pd.unique(actions["game_id"])) if len(actions) else []
    n_matches_in = len(games)
    scored_games, n_excluded = [], 0
    for g in games:
        teams = pd.unique(actions.loc[ids_match(actions["game_id"], g).to_numpy(), "team_id"].dropna())
        if len(teams) == 2:
            scored_games.append(g)
        else:
            n_excluded += 1

    scored_canon = {canonical_id(g) for g in scored_games}
    scored_mask = canonical_id_series(actions["game_id"]).isin(scored_canon).to_numpy()
    a_scored = actions.iloc[scored_mask]

    ctx = add_possession_context(a_scored, params=params)
    spells = build_spells(ctx)
    minutes = possession_minutes(spells)

    pressing = compute_pressing_kpis(ctx, spells, minutes, params=params)
    progression = compute_progression_kpis(ctx, spells, minutes, xg_column=xg_column, params=params)
    buildup = compute_buildup_kpis(ctx, spells, params=params)

    samples = pressing.merge(progression, on=TEAM_KPI_KEYS, how="outer").merge(buildup, on=TEAM_KPI_KEYS, how="outer")
    samples = _add_post_recovery_companions(samples, ctx, spells, xg_column, params)
    samples = _finalize(samples)

    n_shots, n_with_xg, n_null_xg = _shot_census(a_scored, xg_column)
    report = TeamKpiReport(
        params=params,
        n_matches_in=n_matches_in,
        n_matches_scored=len(scored_games),
        n_matches_excluded_not_two_teams=n_excluded,
        n_shots=n_shots,
        n_shots_with_xg=n_with_xg,
        n_shots_null_xg=n_null_xg,
    )
    return samples, report


def _shot_census(a_scored: pd.DataFrame, xg_column: str | None) -> tuple[int, int, int]:
    if len(a_scored) == 0:
        return 0, 0, 0
    is_shot = a_scored["type_id"].isin(_SHOT_TYPE_IDS).to_numpy()
    n_shots = int(is_shot.sum())
    if xg_column is not None and xg_column in a_scored.columns:
        with_xg = int((is_shot & a_scored[xg_column].notna().to_numpy()).sum())
    else:
        with_xg = 0
    return n_shots, with_xg, n_shots - with_xg


def _post_recovery_mask(ctx: pd.DataFrame, spells: pd.DataFrame, window: float) -> np.ndarray:
    """Boolean mask: action is by the recovering team within ``window`` s of one of its recoveries."""
    mask = np.zeros(len(ctx), dtype=bool)
    recov = spells[spells["is_recovery"]]
    if recov.empty or len(ctx) == 0:
        return mask

    def _ck(key):
        return tuple(canonical_id(k) for k in key)

    rstarts = {
        _ck(key): np.sort(grp["start_time"].to_numpy(dtype="float64"))
        for key, grp in recov.groupby(["game_id", "period_id", "team_id"], sort=False)
    }
    ctx_reset = ctx.reset_index(drop=True)
    times = ctx_reset["time_seconds"].to_numpy(dtype="float64")
    for key, idx in ctx_reset.groupby(["game_id", "period_id", "team_id"], sort=False).indices.items():
        starts = rstarts.get(_ck(key))
        if starts is None:
            continue
        t = times[idx]
        j = np.searchsorted(starts, t, side="right") - 1  # most recent recovery start <= t
        ok = j >= 0
        within = np.zeros(len(t), dtype=bool)
        within[ok] = (t[ok] - starts[j[ok]]) <= window
        mask[idx] = within
    return mask


def _add_post_recovery_companions(
    samples: pd.DataFrame, ctx: pd.DataFrame, spells: pd.DataFrame, xg_column: str | None, params
) -> pd.DataFrame:
    has_xg = xg_column is not None and xg_column in ctx.columns
    for src in POST_RECOVERY_SOURCE_COLUMNS:
        samples[f"{src}_post_recovery"] = np.nan

    if len(ctx) > 0:
        pr = ctx.iloc[_post_recovery_mask(ctx, spells, params.post_recovery_window_seconds)]
        if len(pr) > 0:
            tx = pr["start_x"].to_numpy(dtype="float64")
            ty = pr["start_y"].to_numpy(dtype="float64")
            ex = pr["end_x"].to_numpy(dtype="float64")
            ttype = pr["type_id"].to_numpy()

            entered = (tx < _FT_MIN_X) & (ex >= _FT_MIN_X)
            fte = pr.loc[entered, ["game_id", "team_id"]].groupby(["game_id", "team_id"]).size()
            samples = _merge_count(samples, fte, "final_third_entries_post_recovery")

            in_box = (tx >= _BOX_MIN_X) & (np.abs(ty - _FW / 2.0) <= _BOX_HALF_W)
            box = pr.loc[in_box, ["game_id", "team_id"]].groupby(["game_id", "team_id"]).size()
            samples = _merge_count(samples, box, "box_touches_post_recovery")

            is_shot = np.isin(ttype, _SHOT_TYPE_IDS)
            sh = pr.loc[is_shot, ["game_id", "team_id"]].groupby(["game_id", "team_id"]).size()
            samples = _merge_count(samples, sh, "shots_post_recovery")

            if has_xg:
                is_np = np.isin(ttype, _NON_PENALTY_SHOT_TYPE_IDS)
                hi_mask = is_np & (pr[xg_column].to_numpy(dtype="float64") > params.high_opportunity_xg)
                hi = pr.loc[hi_mask, ["game_id", "team_id"]].groupby(["game_id", "team_id"]).size()
                samples = _merge_count(samples, hi, "high_opportunity_shots_post_recovery")

    # the raw counts are real -> absent means 0; high-opp is 0 only when xG is present (else NaN).
    for col in ("final_third_entries_post_recovery", "box_touches_post_recovery", "shots_post_recovery"):
        samples[col] = samples[col].fillna(0)
    if has_xg:
        samples["high_opportunity_shots_post_recovery"] = samples["high_opportunity_shots_post_recovery"].fillna(0)
    return samples


def _merge_count(samples: pd.DataFrame, counts: pd.Series, col: str) -> pd.DataFrame:
    if counts.empty:
        return samples
    c = counts.rename(col).reset_index()
    merged = samples.drop(columns=[col]).merge(c, on=TEAM_KPI_KEYS, how="left")
    return merged


def _finalize(samples: pd.DataFrame) -> pd.DataFrame:
    """Order to TEAM_KPI_COLUMNS and cast dtypes (counts -> Int64, rates -> float64)."""
    if len(samples) == 0:
        return pd.DataFrame({c: pd.Series([], dtype=d) for c, d in TEAM_KPI_COLUMNS.items()})
    out = samples.copy()
    for col in TEAM_KPI_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[list(TEAM_KPI_COLUMNS)]
    return out.astype(TEAM_KPI_COLUMNS).reset_index(drop=True)
