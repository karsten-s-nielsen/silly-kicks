"""``compute_win_probability`` + ``goal_leverage`` -- the per-action public surface.

Per action, the pre-action win/draw/loss and the goal leverage ``ΔP(win | goal)``. Uses ONE
backward-DP outcome table per ``(game, team, man_advantage)`` value (SPEC-12) with O(1) per-action
lookups. Pure -- never mutates ``actions``. Conservation + honest-NaN per ADR-042 / ADR-027.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from silly_kicks.id_compat import canonical_id_series

from ._columns import WIN_PROBABILITY_COLUMNS
from ._config import WinProbabilityParams
from ._model import WinProbabilityModel
from ._report import WinProbabilityReport
from ._state import derive_match_states

_DEFAULT_PARAMS = WinProbabilityParams.default()


def _strength_map(actions: pd.DataFrame, strength_column: str | None) -> dict:
    """{(canonical game, team): base_strength} from a per-row strength column (constant per team)."""
    out: dict = {}
    if strength_column is None or strength_column not in actions.columns:
        return out
    cg = canonical_id_series(actions["game_id"])
    for (g, t), sub in actions.assign(_cg=cg.to_numpy()).groupby(["_cg", "team_id"], sort=False):
        vals = sub[strength_column].dropna()
        if len(vals):
            out[(g, t)] = float(vals.iloc[0])
    return out


def _score_states(states: pd.DataFrame, model: WinProbabilityModel, strength: dict, params: WinProbabilityParams):
    """Return per-state (p_win, p_draw, p_loss, leverage) via one table per (game, team, man_advantage)."""
    K = params.lattice_pad
    step = params.interval_minutes
    n = len(states)
    pw = np.full(n, np.nan)
    pd_ = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    lev = np.full(n, np.nan)
    cg = canonical_id_series(states["game_id"]).to_numpy()
    grp = states.assign(_cg=cg).groupby(["_cg", "team_id", "man_advantage"], sort=False)
    for (g, t, _man_adv_key), sub in grp:
        home = bool(sub["home"].iloc[0])
        bs = strength.get((g, t), 0.0)
        man_adv = int(sub["man_advantage"].to_numpy()[0])
        max_rem = float(sub["minutes_remaining"].max())
        n_tab = max(math.ceil(max_rem / step), 0)
        Pw, Pd, Pl = model.outcome_table(base_strength=bs, home=home, man_advantage=man_adv, n_intervals=n_tab)
        for pos, (_, r) in zip(sub["_pos"].to_numpy(), sub.iterrows(), strict=False):
            m = max(math.ceil(float(r["minutes_remaining"]) / step), 0)
            di = int(np.clip(int(r["score_diff"]) + K, 0, 2 * K))  # clamp extreme blowouts to the pad
            w_raw, d_raw, l_raw = float(Pw[di, m]), float(Pd[di, m]), float(Pl[di, m])
            w = model._apply_isotonic(w_raw)
            rem = max(1.0 - w, 0.0)
            dl = d_raw + l_raw
            d_cal = rem * (d_raw / dl) if dl > 0 else rem / 2.0
            di_up = min(di + 1, 2 * K)
            leverage = model._apply_isotonic(float(Pw[di_up, m])) - w
            pw[pos] = w
            pd_[pos] = d_cal
            pl[pos] = rem - d_cal
            lev[pos] = leverage
    return pw, pd_, pl, lev


def compute_win_probability(
    actions: pd.DataFrame,
    *,
    model: WinProbabilityModel,
    games: pd.DataFrame | None = None,
    strength_column: str | None = None,
    params: WinProbabilityParams = _DEFAULT_PARAMS,
) -> tuple[pd.DataFrame, WinProbabilityReport]:
    """Per-action win/draw/loss + goal leverage. Returns ``(samples, report)``.

    Examples
    --------
    A fitted :class:`~silly_kicks.win_probability.WinProbabilityModel` scores each action's pre-action
    game state (a real ``actions`` frame + a ``(game_id, home_team_id)`` frame are required, so this is
    an illustrative block)::

        from silly_kicks.win_probability import WinProbabilityModel, compute_win_probability

        samples, report = compute_win_probability(actions, model=WinProbabilityModel.bundled(), games=games)
        samples[["p_win", "win_prob_leverage"]].head()
    """
    if len(actions) == 0:
        empty = pd.DataFrame({c: pd.Series([], dtype=d) for c, d in WIN_PROBABILITY_COLUMNS.items()})
        return empty, WinProbabilityReport(params, 0, 0, 0, 0, 0)

    cg = canonical_id_series(actions["game_id"])
    nteams = (
        pd.DataFrame({"_cg": cg.to_numpy(), "team_id": actions["team_id"].to_numpy()})
        .groupby("_cg", sort=False)["team_id"]
        .nunique(dropna=True)
    )
    n_matches_in = int(nteams.shape[0])
    scored_canon = set(nteams.index[nteams == 2])
    n_excluded = int((nteams != 2).sum())

    scored_mask = cg.isin(scored_canon).to_numpy()
    strength = _strength_map(actions, strength_column)

    rows = pd.DataFrame(
        {
            "game_id": actions["game_id"].to_numpy(),
            "action_id": actions["action_id"].to_numpy().astype("int64"),
            "team_id": actions["team_id"].to_numpy(),
            "period_id": actions["period_id"].to_numpy().astype("int64"),
            "p_win": np.nan,
            "p_draw": np.nan,
            "p_loss": np.nan,
            "win_prob_leverage": np.nan,
            "win_prob_source": np.where(scored_mask, "scored", "excluded_not_two_teams"),
        }
    )

    n_actions_unresolved = 0
    if scored_mask.any():
        a_scored = actions.iloc[scored_mask].copy()
        states = derive_match_states(a_scored, games=games, params=params)
        states = states.reset_index(drop=True)
        states["_pos"] = np.arange(len(states))
        resolved = states["state_source"].to_numpy() == "resolved"
        pw, pd_, pl, lev = _score_states(states, model, strength, params)
        # map state rows back to `rows` by (canonical game_id, action_id)
        rows_cg = canonical_id_series(rows["game_id"]).to_numpy()
        rows_aid = rows["action_id"].to_numpy().tolist()
        state_cg = canonical_id_series(states["game_id"]).to_numpy()
        state_aid = states["action_id"].to_numpy().tolist()
        pos_of = {(g, a): i for i, (g, a) in enumerate(zip(state_cg, state_aid, strict=False))}
        n_rows = len(rows)
        pw_col = np.full(n_rows, np.nan)
        pd_col = np.full(n_rows, np.nan)
        pl_col = np.full(n_rows, np.nan)
        lev_col = np.full(n_rows, np.nan)
        src_col = rows["win_prob_source"].to_numpy().copy()
        for ri, (g, a) in enumerate(zip(rows_cg, rows_aid, strict=False)):
            sp = pos_of.get((g, a))
            if sp is None:
                continue
            if not resolved[sp]:
                src_col[ri] = "unresolved_state"
                n_actions_unresolved += 1
                continue
            pw_col[ri] = pw[sp]
            pd_col[ri] = pd_[sp]
            pl_col[ri] = pl[sp]
            lev_col[ri] = lev[sp]
        rows["p_win"] = pw_col
        rows["p_draw"] = pd_col
        rows["p_loss"] = pl_col
        rows["win_prob_leverage"] = lev_col
        rows["win_prob_source"] = src_col

    samples = rows.astype(WIN_PROBABILITY_COLUMNS).reset_index(drop=True)
    report = WinProbabilityReport(
        params=params,
        n_matches_in=n_matches_in,
        n_matches_scored=len(scored_canon),
        n_matches_excluded_not_two_teams=n_excluded,
        n_actions=len(actions),
        n_actions_unresolved=n_actions_unresolved,
    )
    return samples, report


def goal_leverage(
    actions: pd.DataFrame,
    *,
    model: WinProbabilityModel,
    games: pd.DataFrame | None = None,
    strength_column: str | None = None,
    params: WinProbabilityParams = _DEFAULT_PARAMS,
) -> pd.Series:
    """``ΔP(win | goal)`` per action, aligned to ``actions.index`` (honest-NaN where unresolved).

    Examples
    --------
    The per-action goal leverage, ready to weight ``VAEP_adjusted`` into xImpact (a real ``actions``
    frame + a fitted model are required, so this is an illustrative block)::

        from silly_kicks.win_probability import WinProbabilityModel, goal_leverage

        leverage = goal_leverage(actions, model=WinProbabilityModel.bundled(), games=games)
        leverage.describe()
    """
    samples, _ = compute_win_probability(
        actions, model=model, games=games, strength_column=strength_column, params=params
    )
    # samples row order matches actions row order (built from actions.to_numpy()); align by position.
    return pd.Series(samples["win_prob_leverage"].to_numpy(), index=actions.index, name="win_prob_leverage")
