"""Observation-level -> per-keeper aggregation (spec §5).

Keyed on the frames-resolved GK ``player_id``: a pure library module must not depend on the
gold-mart ``player_key`` join.
"""

from __future__ import annotations

import pandas as pd

#: Columns the aggregate always carries, in emission order. Pinning the projection (rather
#: than returning whatever the groupby happened to build) is what makes the schema stable
#: across an EMPTY and a populated input -- a caller that filters to zero rows gets the same
#: columns AND the same dtypes as one that does not.
_OUTPUT_COLUMNS = ["player_id", "mean", "median", "n", "n_nonzero", "n_games", "gate_eligible"]


def aggregate_by_keeper(
    observations: pd.DataFrame,
    *,
    value_col: str,
    min_nonzero: int = 20,
    min_games: int = 2,
) -> pd.DataFrame:
    """Aggregate observation-level arm values to per-keeper rows.

    Keyed on the frames-resolved GK ``player_id`` (spec §5): every frame row carries one,
    and the gold-mart ``player_key`` is deliberately NOT used -- it is an actions-grain
    lakehouse column, and a pure library module must not depend on a gold join.

    Grain-agnostic: any observation-level table with ``player_id``, ``game_id`` and
    ``value_col`` aggregates, so a future window-grain arm reuses this unchanged.

    Reports mean AND median (the registered gate reads the mean; the median is the
    outlier-robust companion), plus per-keeper **nonzero** counts -- ΔDAS is exactly 0
    whenever the displacement moves no accessible-space boundary, and small displacements
    dominate, so a raw ``n`` overstates how much evidence a keeper actually contributed.

    ``gate_eligible`` encodes the spec §6.1 registered clustering floors: ``min_nonzero``
    informative observations AND ``min_games`` distinct matches. The second floor is not
    cosmetic -- for a single-match keeper, keeper ≡ match, so between-keeper variance
    mechanically absorbs between-match variance and the ICC is inflated by construction.

    A **NaN value is never counted as nonzero**. ``value != 0`` is True for NaN, so the
    obvious count would let an all-NaN keeper clear the registered floor -- exactly the
    silent-null failure class this cycle exists to eliminate. NaN rows still count toward
    ``n`` (they were sampled) and are skipped by ``mean``/``median`` (pandas default), so
    ``n`` is the sampled grain and ``n_nonzero`` is the informative one.

    Parameters
    ----------
    observations : pd.DataFrame
        Observation-level values with ``player_id``, ``game_id`` and ``value_col``.
        Never mutated.
    value_col : str
        Column holding the arm value to aggregate.
    min_nonzero, min_games : int
        Registered eligibility floors, echoed into ``gate_eligible``.

    Returns
    -------
    pd.DataFrame
        One row per keeper with ``mean``, ``median``, ``n``, ``n_nonzero``, ``n_games``
        and ``gate_eligible``.

    Examples
    --------
    >>> obs = pd.DataFrame(
    ...     {"player_id": [1, 1], "game_id": ["g1", "g2"], "delta_das": [-0.5, 0.0]}
    ... )
    >>> aggregate_by_keeper(obs, value_col="delta_das", min_nonzero=1)["n_nonzero"].tolist()
    [1]
    """
    src = observations  # never mutated
    missing = {"player_id", "game_id", value_col} - set(src.columns)
    if missing:
        raise ValueError(f"aggregate_by_keeper: observations is missing {sorted(missing)}")

    # NOTE, deliberately no empty-input special case: the groupby path already returns the
    # declared schema with the SAME dtypes on zero rows (measured), and a hand-rolled empty
    # branch was strictly worse -- it had to guess dtypes and got the counts wrong (float64
    # instead of int64), so an empty result would not concatenate cleanly with a populated
    # one. The empty contract is covered behaviourally instead.
    grp = src.groupby("player_id", dropna=True)
    out = grp.agg(
        mean=(value_col, "mean"),
        median=(value_col, "median"),
        n=(value_col, "size"),
        n_games=("game_id", "nunique"),
    ).reset_index()

    # NaN-safe: informative == present AND non-zero. Grouping the boolean Series by the
    # player_id Series keeps this vectorized and keeps `src` untouched (no assign/copy).
    informative = src[value_col].notna() & src[value_col].ne(0)
    nz = informative.groupby(src["player_id"], dropna=True).sum().rename("n_nonzero").reset_index()
    out = out.merge(nz, on="player_id", how="left")
    # `how="left"` cannot actually miss (both sides group the same column of the same frame),
    # but the fillna keeps a future merge-key change from silently producing NaN counts that
    # would compare False against every floor -- i.e. fail CLOSED, not open.
    out["n_nonzero"] = out["n_nonzero"].fillna(0).astype(int)

    out["gate_eligible"] = (out["n_nonzero"] >= min_nonzero) & (out["n_games"] >= min_games)
    return out[_OUTPUT_COLUMNS]
