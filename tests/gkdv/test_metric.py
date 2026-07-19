"""Per-keeper aggregation: grain-agnostic, keyed on the frames-resolved GK player_id."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.gkdv import aggregate_by_keeper


def _obs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [10, 10, 10, 20, 20],
            "game_id": ["g1", "g1", "g2", "g1", "g2"],
            "value": [-0.02, -0.01, 0.0, 0.03, 0.0],
        }
    )


def test_reports_mean_AND_median_and_nonzero_counts():
    out = aggregate_by_keeper(_obs(), value_col="value", min_nonzero=1)
    assert {"player_id", "mean", "median", "n", "n_nonzero", "n_games"} <= set(out.columns)
    row = out[out["player_id"] == 10].iloc[0]
    assert row["n"] == 3 and row["n_nonzero"] == 2
    # Assert the VALUES, not merely the columns: a column present but mis-wired (median
    # aliased to mean, n_games aliased to n) would pass a presence-only check.
    assert row["mean"] == pytest.approx(-0.01)
    assert row["median"] == pytest.approx(-0.01)
    assert row["n_games"] == 2


def test_mean_and_median_differ_on_a_skewed_keeper():
    """Both statistics are reported because they disagree under skew (the registered gate
    reads the mean; the median is the outlier-robust companion)."""
    obs = pd.DataFrame(
        {
            "player_id": [30] * 5,
            "game_id": ["g1", "g1", "g2", "g2", "g3"],
            "value": [-0.01, -0.01, -0.01, -0.01, -5.0],
        }
    )
    row = aggregate_by_keeper(obs, value_col="value", min_nonzero=1).iloc[0]
    assert row["median"] == pytest.approx(-0.01)
    assert row["mean"] < -0.9  # the outlier moves the mean but not the median


def test_min_nonzero_excludes_a_keeper_from_the_gate_surface():
    out = aggregate_by_keeper(_obs(), value_col="value", min_nonzero=2)
    eligible = set(out.loc[out["gate_eligible"], "player_id"])
    assert 20 not in eligible
    # ...and the gate is not vacuously empty -- keeper 10 DOES clear it.
    assert 10 in eligible


def test_min_games_excludes_a_single_match_keeper():
    """Spec 6.1 clustering floor: for a single-match keeper, keeper == match, so
    between-keeper variance mechanically absorbs between-match variance."""
    obs = pd.DataFrame(
        {
            "player_id": [40, 40, 50, 50],
            "game_id": ["g1", "g1", "g1", "g2"],
            "value": [0.1, 0.2, 0.1, 0.2],
        }
    )
    out = aggregate_by_keeper(obs, value_col="value", min_nonzero=1, min_games=2)
    eligible = set(out.loc[out["gate_eligible"], "player_id"])
    assert 40 not in eligible  # both observations in g1
    assert 50 in eligible  # spans g1 and g2


def test_nan_is_not_counted_as_a_nonzero_observation():
    """A NaN arm value is NOT evidence of a moved boundary.

    ``value != 0`` is True for NaN, so a naive nonzero count would let an all-NaN keeper
    clear the registered floor -- the silent-null failure class this cycle exists to
    eliminate.
    """
    obs = pd.DataFrame(
        {
            "player_id": [60] * 4,
            "game_id": ["g1", "g1", "g2", "g2"],
            "value": [np.nan, np.nan, np.nan, np.nan],
        }
    )
    out = aggregate_by_keeper(obs, value_col="value", min_nonzero=1)
    assert int(out.iloc[0]["n_nonzero"]) == 0
    assert not bool(out.iloc[0]["gate_eligible"])


def test_input_is_not_mutated():
    df = _obs()
    before = df.copy(deep=True)
    aggregate_by_keeper(df, value_col="value", min_nonzero=1)
    pd.testing.assert_frame_equal(df, before)


def test_empty_input_returns_the_SAME_schema_as_a_populated_one():
    """An empty result must concatenate cleanly with a populated one.

    Asserting only "no crash, columns present" is vacuous -- the natural groupby path
    already satisfies that. What can actually break is a well-meant empty-input special
    case that has to GUESS dtypes: guess float for the counts and every downstream
    ``pd.concat`` silently upcasts. So compare dtypes against the populated path.
    """
    populated = aggregate_by_keeper(_obs(), value_col="value", min_nonzero=1)
    out = aggregate_by_keeper(_obs().iloc[:0], value_col="value", min_nonzero=1)
    assert len(out) == 0
    assert list(out.columns) == list(populated.columns)
    pd.testing.assert_series_equal(out.dtypes, populated.dtypes)
    # ...and the round trip is what the dtype parity is FOR.
    assert list(pd.concat([out, populated], ignore_index=True).dtypes) == list(populated.dtypes)


def test_aggregation_is_grain_agnostic():
    """Any observation-level table aggregates -- a future window-grain arm reuses this
    unchanged, so the function must not assume a frame grain."""
    obs = _obs().rename(columns={"value": "delta_das"})
    out = aggregate_by_keeper(obs, value_col="delta_das", min_nonzero=1)
    assert set(out["player_id"]) == {10, 20}
