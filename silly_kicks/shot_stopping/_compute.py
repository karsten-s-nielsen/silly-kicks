"""compute_shot_stopping -- Goals Prevented / GSAA per (goalkeeper, match) (TF-59 PR2, spec §6).

Event-only. Consumes SPADL ``actions`` + an INJECTED ``psxg_column`` + the PR1-stamped
``defending_gk_column`` / ``defending_team_column`` (keeper_identity.add_defending_gk_player_id).
Output ids are RAW (the keeper id from ``defending_gk_player_id`` and its authoritative team from
``defending_gk_team_id``), so ``player_id`` / ``team_id`` match the resolver's raw ids -- consumers
join via ``id_compat`` (ADR-019) if their dim tables use another representation. Keepers are GROUPED
on their CANONICAL id (ADR-019) so a keeper whose raw id appears in mixed dtypes across a match is not
fragmented into two split-GSAA rows; the raw id is emitted via ``.first()``. PURE -- never mutates
``actions``. See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import pandas as pd

from silly_kicks.id_compat import canonical_id_series
from silly_kicks.spadl import config as spadlconfig

from ._columns import (
    SHOT_STOPPING_COLUMNS,
    SS_GOALS_CONCEDED,
    SS_GOALS_CONCEDED_EXCL_PEN,
    SS_GOALS_PREVENTED,
    SS_GOALS_PREVENTED_EXCL_PEN,
    SS_PSXG_FACED,
    SS_PSXG_FACED_EXCL_PEN,
    SS_SHOTS_FACED,
    SS_SHOTS_FACED_EXCL_PEN,
    SS_TEAM_ID,
)
from ._config import ShotStoppingParams
from ._report import ShotStoppingReport

_DEFAULT_PARAMS = ShotStoppingParams()  # module-level singleton (avoids a B008 call-in-default)

_SHOT_TYPE_IDS = frozenset(spadlconfig.actiontype_id[n] for n in ("shot", "shot_penalty", "shot_freekick"))
_PENALTY_TYPE_ID = spadlconfig.actiontype_id["shot_penalty"]
_SUCCESS = spadlconfig.result_id["success"]


def compute_shot_stopping(
    actions: pd.DataFrame,
    *,
    psxg_column: str,
    defending_gk_column: str = "defending_gk_player_id",
    defending_team_column: str = "defending_gk_team_id",
    params: ShotStoppingParams = _DEFAULT_PARAMS,
) -> tuple[pd.DataFrame, ShotStoppingReport]:
    """Goals Prevented / GSAA per (game_id, defending keeper player_id). Returns (samples, report).

    Examples
    --------
    Stamp the defending keeper + team (PR1) then compute Goals Prevented / GSAA per keeper from an
    injected Post-Shot xG column (``tests/shot_stopping/test_compute.py`` has the worked numbers)::

        from silly_kicks.keeper_identity import add_defending_gk_player_id
        from silly_kicks.shot_stopping import compute_shot_stopping

        stamped = add_defending_gk_player_id(actions, keeper_map, appearances=appearances)
        samples, report = compute_shot_stopping(stamped, psxg_column="post_shot_xg")
    """
    if psxg_column not in actions.columns:
        raise KeyError(
            f"psxg_column {psxg_column!r} is not in actions -- silly-kicks ships no xG/PSxG model. "
            "Inject a Post-Shot xG column (port pattern; cf. xg_column in vaep/labels.py, ADR-085)."
        )
    for col in (defending_gk_column, defending_team_column):
        if col not in actions.columns:
            raise KeyError(
                f"{col!r} is not in actions -- stamp it first with "
                "silly_kicks.keeper_identity.add_defending_gk_player_id (ADR-084/085)."
            )

    psxg = pd.to_numeric(actions[psxg_column], errors="coerce")
    is_shot = actions["type_id"].isin(_SHOT_TYPE_IDS)
    on_target = psxg.notna()  # PSxG presence IS the on-target gate
    not_blocked = ~actions["shot_blocked"].eq(True).fillna(False)  # NA -> not blocked (ADR-046)
    not_shootout = actions["period_id"].ne(params.shootout_period_id)
    # Own goals are bad_touch+owngoal (ADR-018), so is_shot ALREADY excludes them -- no separate mask.
    faced_mask = is_shot & on_target & not_blocked & not_shootout

    faced = pd.DataFrame(
        {
            "game_id": actions["game_id"],
            "keeper": actions[defending_gk_column],  # RAW keeper id (as the resolver stamped it)
            "team": actions[defending_team_column],  # RAW authoritative defending team (ADR-085)
            "psxg": psxg,
            "is_goal": (actions["result_id"] == _SUCCESS),
            "is_penalty": (actions["type_id"] == _PENALTY_TYPE_ID),
        }
    )[faced_mask.to_numpy()].reset_index(drop=True)
    # CANONICAL group keys (ADR-019): grouping on the RAW keeper id fragments a keeper whose id appears
    # in mixed dtypes across the match (int 88 vs str "88" -> two rows, split GSAA). Group on the
    # canonical keys; emit the RAW game_id/keeper/team via .first() so OUTPUT ids stay raw (ADR-085).
    faced["_gkey"] = canonical_id_series(faced["game_id"])
    faced["_kkey"] = canonical_id_series(faced["keeper"])

    n_faced = len(faced)
    attributed = faced[faced["keeper"].notna()]
    n_attr = len(attributed)
    report = ShotStoppingReport(
        params=params,
        n_shots_faced=n_faced,
        n_shots_attributed=n_attr,
        n_shots_unattributed=n_faced - n_attr,
    )

    empty = pd.DataFrame({c: pd.Series(dtype=t) for c, t in SHOT_STOPPING_COLUMNS.items()})
    if attributed.empty:
        return empty, report

    def _agg(frame: pd.DataFrame) -> pd.DataFrame:
        # Group on the CANONICAL keys (dtype-safe, ADR-019); emit the RAW game_id/keeper/team via
        # .first() so a keeper's output ids keep their provider representation (ADR-085).
        g = frame.groupby(["_gkey", "_kkey"], dropna=True, sort=False)
        out = pd.DataFrame(
            {
                "game_id": g["game_id"].first(),
                "keeper": g["keeper"].first(),
                "team": g["team"].first(),
                "shots_faced": g.size(),
                "goals_conceded": g["is_goal"].sum(),
                "psxg_faced": g["psxg"].sum(min_count=1),
            }
        )
        out["goals_prevented"] = out["psxg_faced"] - out["goals_conceded"]
        return out.reset_index()  # _gkey, _kkey become the merge keys

    full = _agg(attributed).rename(
        columns={
            "shots_faced": SS_SHOTS_FACED,
            "goals_conceded": SS_GOALS_CONCEDED,
            "psxg_faced": SS_PSXG_FACED,
            "goals_prevented": SS_GOALS_PREVENTED,
        }
    )
    excl = (
        _agg(attributed[~attributed["is_penalty"]])
        .rename(
            columns={
                "shots_faced": SS_SHOTS_FACED_EXCL_PEN,
                "goals_conceded": SS_GOALS_CONCEDED_EXCL_PEN,
                "psxg_faced": SS_PSXG_FACED_EXCL_PEN,
                "goals_prevented": SS_GOALS_PREVENTED_EXCL_PEN,
            }
        )
        .drop(columns=["game_id", "keeper", "team"])  # raw ids come from `full`; merge on canonical keys
    )

    merged = full.merge(excl, on=["_gkey", "_kkey"], how="left")
    merged[SS_TEAM_ID] = merged["team"]  # AUTHORITATIVE keeper team (from defending_gk_team_id)
    out = merged.rename(columns={"keeper": "player_id"}).reindex(columns=list(SHOT_STOPPING_COLUMNS))
    # A keeper who faced only penalties has no open-play rows -> the excl companions are NA from the left
    # merge; a keeper with 0 non-penalty shots has 0 excl-penalty shots, GP 0.0.
    for col, fill in (
        (SS_SHOTS_FACED_EXCL_PEN, 0),
        (SS_GOALS_CONCEDED_EXCL_PEN, 0),
        (SS_PSXG_FACED_EXCL_PEN, 0.0),
        (SS_GOALS_PREVENTED_EXCL_PEN, 0.0),
    ):
        out[col] = out[col].fillna(fill)
    return out.astype(SHOT_STOPPING_COLUMNS), report
