"""``compute_match_outcome`` -- the TF-53 orchestrator.

Per ``(game_id, team_id)``: filter shots, build each team's exact Poisson-binomial goal PMF, and sum
the two-team outcome simplex into ``p_win`` / ``p_draw`` / ``p_loss`` / ``xpoints``. Pure: never mutates
``actions``. A game whose actions carry != 2 distinct team ids is excluded-and-counted (ADR-042); own
goals are counted (result-only, ADR-018) for scoreline reconciliation but carry no xG (spec §7).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks._frame_index import group_rows
from silly_kicks.id_compat import canonical_id_series, ids_match
from silly_kicks.spadl import config as spadlconfig

from ._columns import MATCH_OUTCOME_COLUMNS
from ._config import MatchOutcomeParams
from ._pmf import goal_count_pmf, match_outcome_probabilities
from ._report import MatchOutcomeReport

_SHOT_TYPE_IDS = [spadlconfig.actiontype_id[n] for n in ("shot", "shot_penalty", "shot_freekick")]
_OWNGOAL = spadlconfig.result_id["owngoal"]
_DEFAULT_PARAMS = MatchOutcomeParams.default()


def compute_match_outcome(
    actions: pd.DataFrame,
    *,
    xg_column: str,
    params: MatchOutcomeParams = _DEFAULT_PARAMS,
) -> tuple[pd.DataFrame, MatchOutcomeReport]:
    """Per-``(game_id, team_id)`` win/draw/loss probabilities + xPoints from an injected per-shot xG.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions (``type_id`` / ``result_id`` int-coded) for one or more matches.
    xg_column : str
        Name of the injected pre-shot xG column (silly-kicks ships no xG -- port pattern).
    params : MatchOutcomeParams
        Frozen thresholds; the two opt-in corrections (spec §4).

    Returns
    -------
    (samples, report)
        ``samples``: one row per ``(game_id, team_id)`` over ``MATCH_OUTCOME_COLUMNS``.
        ``report``: a :class:`MatchOutcomeReport` conserving the match + shot-xG population.

    Examples
    --------
    Injected per-shot xG (e.g. StatsBomb's ``statsbomb_xg``) yields per-team outcome probabilities::

        samples, report = compute_match_outcome(actions, xg_column="xg")
        home_xp = samples.loc[samples["team_id"] == home_team_id, "xpoints"]
    """
    if len(actions) == 0:
        return _finalize([]), MatchOutcomeReport(params, 0, 0, 0, 0, 0, 0, 0)

    # Distinct-team count per game in ONE vectorized groupby (never a per-game rescan; ADR-068).
    canon = canonical_id_series(actions["game_id"])
    nteams = (
        pd.DataFrame({"_cg": canon.to_numpy(), "team_id": actions["team_id"].to_numpy()})
        .groupby("_cg", sort=False)["team_id"]
        .nunique(dropna=True)
    )
    n_matches_in = int(nteams.shape[0])
    scored_canon = set(nteams.index[nteams == 2])
    n_excluded = int((nteams != 2).sum())

    scored_mask = canon.isin(scored_canon).to_numpy()
    a_scored = actions.iloc[scored_mask]

    # add_possessions only when collapsing (Rung 3b); the independent path never touches it.
    ctx = a_scored
    if params.same_possession == "collapse" and len(a_scored):
        from silly_kicks import spadl

        ctx = spadl.add_possessions(a_scored, max_gap_seconds=params.possession_max_gap_seconds)

    game_groups = group_rows(ctx, "game_id")
    rows: list[dict] = []
    for game_id in pd.unique(a_scored["game_id"]):
        ga = game_groups.get(game_id)
        teams = list(pd.unique(ga["team_id"].dropna()))
        pmfs = [_team_pmf(ga.iloc[ids_match(ga["team_id"], t).to_numpy()], xg_column, params) for t in teams]
        for idx, t in enumerate(teams):  # exactly two scored teams -> opponent is the other row
            own, opp = pmfs[idx], pmfs[1 - idx]
            p_win, p_draw, p_loss = match_outcome_probabilities(own, opp, params=params)
            ta = ga.iloc[ids_match(ga["team_id"], t).to_numpy()]
            eg = float(ta.loc[ta["type_id"].isin(_SHOT_TYPE_IDS), xg_column].sum()) if xg_column in ta.columns else 0.0
            rows.append(
                {
                    "game_id": game_id,
                    "team_id": t,
                    "p_win": p_win,
                    "p_draw": p_draw,
                    "p_loss": p_loss,
                    "xpoints": 3.0 * p_win + p_draw,
                    "expected_goals": eg,
                }
            )

    samples = _finalize(rows)
    n_shots, n_with_xg, n_null_xg, n_own = _census(a_scored, xg_column)
    report = MatchOutcomeReport(
        params=params,
        n_matches_in=n_matches_in,
        n_matches_scored=len(scored_canon),
        n_matches_excluded_not_two_teams=n_excluded,
        n_shots=n_shots,
        n_shots_with_xg=n_with_xg,
        n_shots_null_xg=n_null_xg,
        n_own_goals=n_own,
    )
    return samples, report


def _team_pmf(team_shots_and_actions: pd.DataFrame, xg_column: str, params: MatchOutcomeParams) -> np.ndarray:
    """One team's exact goal PMF over its shot xGs (NaN-xg shots dropped; Rung-3b collapse if set)."""
    shots = team_shots_and_actions[team_shots_and_actions["type_id"].isin(_SHOT_TYPE_IDS)]
    if xg_column not in shots.columns:  # honest: no xG at all -> every team scores 0 (P(0)=1)
        return goal_count_pmf([])
    if params.same_possession == "collapse":
        from ._collapse import collapse_team_xgs

        xgs = collapse_team_xgs(shots, xg_column=xg_column)
    else:
        xgs = shots[xg_column].dropna().to_numpy(dtype="float64")
    return goal_count_pmf(xgs)


def _census(a_scored: pd.DataFrame, xg_column: str) -> tuple[int, int, int, int]:
    if len(a_scored) == 0:
        return 0, 0, 0, 0
    is_shot = a_scored["type_id"].isin(_SHOT_TYPE_IDS).to_numpy()
    n_shots = int(is_shot.sum())
    with_xg = int((is_shot & a_scored[xg_column].notna().to_numpy()).sum()) if xg_column in a_scored.columns else 0
    n_own = int((a_scored["result_id"] == _OWNGOAL).sum())
    return n_shots, with_xg, n_shots - with_xg, n_own


def _finalize(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame({c: pd.Series([], dtype=d) for c, d in MATCH_OUTCOME_COLUMNS.items()})
    out = pd.DataFrame(rows)[list(MATCH_OUTCOME_COLUMNS)]
    return out.astype(MATCH_OUTCOME_COLUMNS).reset_index(drop=True)
