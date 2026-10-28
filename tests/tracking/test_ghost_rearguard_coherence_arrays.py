"""`ghost_rearguard_coherence` over arrays equals its `2474b73` per-group body (HFU plan R3-8 S3 item 15 (a)).

The per-group pandas loop took 416 profiled seconds of the ghost_outfield `fit` unit on HF (S1g, 1,042,736 groups).
No-rebaseline kind (c): `_coherence_oracle` below is the `2474b73` body VERBATIM, and every case compares the two
results as the trainer writes them (`json.dumps`), so a NaN, an int-vs-float count or a changed float shows up.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._ghost_outfield import ghost_rearguard_coherence


# The `2474b73` `ghost_rearguard_coherence`, VERBATIM (the D20 oracle; only the name differs).
def _coherence_oracle(served: pd.DataFrame) -> dict:
    """Measure whether an independently-predicted ghost rearguard hangs together as a line.

    ``served`` is a :func:`serve_ghost_outfield_positions` output. Reported (spec 9), NOT gated: the
    per-slot model is built independently, and a shape constraint is added only if this measurement
    shows material incoherence. Two metrics per ``(game, period, frame, team)`` group of >= 2 finite
    ghost slots:

    * ``ordering_fraction`` --- the fraction of groups whose ghost slots preserve their lateral rank
      (``ghost_gr_y`` non-decreasing in ``slot_index`` order). 1.0 = every ghost line is well ordered.
    * ``min_pairwise_distance_m`` --- the smallest slot-to-slot distance seen across all groups (a
      collapse indicator).
    """
    finite = served.dropna(subset=["ghost_gr_x", "ghost_gr_y"])
    n_groups = 0
    n_ordered = 0
    min_dists: list[float] = []
    for _key, grp in finite.groupby(["game_id", "period_id", "frame_id", "team_id"], sort=False, observed=True):
        gg = grp.sort_values("slot_index", kind="stable")
        if len(gg) < 2:
            continue
        n_groups += 1
        ys = gg["ghost_gr_y"].to_numpy(dtype=float)
        xs = gg["ghost_gr_x"].to_numpy(dtype=float)
        if np.all(np.diff(ys) >= 0.0):
            n_ordered += 1
        pts = np.column_stack([xs, ys])
        dmat = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=-1))
        np.fill_diagonal(dmat, np.inf)
        min_dists.append(float(dmat.min()))
    return {
        "n_groups": n_groups,
        "ordering_fraction": (n_ordered / n_groups) if n_groups else float("nan"),
        "min_pairwise_distance_m": float(np.min(min_dists)) if min_dists else float("nan"),
    }


def _assert_same(served: pd.DataFrame) -> None:
    want = _coherence_oracle(served)
    got = ghost_rearguard_coherence(served)
    assert json.dumps(got) == json.dumps(want)
    assert [type(v) for v in got.values()] == [type(v) for v in want.values()]


def _random_served(seed: int, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    served = pd.DataFrame(
        {
            "game_id": rng.choice(["g1", "g2"], n),
            "period_id": rng.integers(1, 3, n),
            "frame_id": rng.integers(0, 40, n),
            "team_id": rng.choice([10, 20], n),
            "slot_index": rng.integers(0, 5, n).astype(float),  # ties inside a group
            "ghost_gr_x": rng.uniform(0.0, 105.0, n).round(1),  # equal positions -> zero distances
            "ghost_gr_y": rng.uniform(0.0, 68.0, n).round(1),
        }
    )
    served.loc[rng.random(n) < 0.05, "ghost_gr_x"] = np.nan
    served.loc[rng.random(n) < 0.05, "ghost_gr_y"] = np.nan
    return served


@pytest.mark.parametrize("seed", range(12))
def test_random_frames_match_the_oracle(seed):
    _assert_same(_random_served(seed))


def test_missing_keys_are_dropped_like_the_groupby():
    served = _random_served(101)
    served = served.astype({"game_id": object, "team_id": object})
    served.loc[served.index[::7], "game_id"] = None
    served.loc[served.index[::9], "team_id"] = np.nan
    served.loc[served.index[::13], "period_id"] = pd.NA
    _assert_same(served)


def test_missing_slot_index_sorts_last_and_ties_keep_row_order():
    served = _random_served(202)
    served.loc[served.index[::5], "slot_index"] = np.nan
    _assert_same(served)
    _assert_same(served.iloc[::-1].reset_index(drop=True))  # another row order -> another stable order


def test_infinite_coordinates_propagate_nan_like_the_oracle():
    served = _random_served(303)
    served.loc[served.index[3], "ghost_gr_x"] = np.inf
    served.loc[served.index[4], "ghost_gr_x"] = np.inf
    served.loc[served.index[8], "ghost_gr_y"] = -np.inf
    _assert_same(served)


def test_categorical_keys_and_float32_coordinates():
    served = _random_served(404)
    served["team_id"] = served["team_id"].astype("category").cat.add_categories([99])  # an unobserved category
    served["game_id"] = served["game_id"].astype("category")
    served["ghost_gr_x"] = served["ghost_gr_x"].astype(np.float32)
    _assert_same(served)


def test_mixed_int_and_float_game_ids_group_together():
    served = _random_served(505)
    served["game_id"] = np.where(np.arange(len(served)) % 2 == 0, 1, 1.0).astype(object)
    _assert_same(served)


def test_degenerate_tables():
    cols = ["game_id", "period_id", "frame_id", "team_id", "slot_index", "ghost_gr_x", "ghost_gr_y"]
    _assert_same(pd.DataFrame(columns=cols))  # empty
    single = pd.DataFrame([["g", 1, 1, 1, 0.0, 1.0, 2.0]], columns=cols)
    _assert_same(single)  # one group of one slot -> no group counts
    all_nan = _random_served(606)
    all_nan["ghost_gr_y"] = np.nan
    _assert_same(all_nan)
    one_wide = pd.DataFrame(
        {
            "game_id": ["g"] * 9,
            "period_id": [1] * 9,
            "frame_id": [7] * 9,
            "team_id": [3] * 9,
            "slot_index": [8.0, 1.0, 3.0, 2.0, 0.0, 5.0, 4.0, 7.0, 6.0],
            "ghost_gr_x": np.linspace(10.0, 30.0, 9),
            "ghost_gr_y": np.linspace(5.0, 60.0, 9),
        }
    )
    _assert_same(one_wide)  # a group wider than any rearguard


def test_the_existing_non_vacuity_cases_still_hold():
    ordered = pd.DataFrame(
        {
            "game_id": ["g"] * 4,
            "period_id": [1] * 4,
            "frame_id": [1] * 4,
            "team_id": ["A"] * 4,
            "slot_index": [1.0, 2.0, 3.0, 4.0],
            "ghost_gr_x": [20.0, 20.0, 20.0, 20.0],
            "ghost_gr_y": [10.0, 25.0, 40.0, 55.0],
        }
    )
    _assert_same(ordered)
    shuffled = ordered.assign(ghost_gr_y=[55.0, 40.0, 25.0, 10.0])
    _assert_same(shuffled)
    assert ghost_rearguard_coherence(shuffled)["ordering_fraction"] == 0.0


def test_no_per_group_python_loop(monkeypatch):
    """The array body never iterates the groupby (the loop is what cost 416 s on HF)."""
    from pandas.core.groupby.generic import DataFrameGroupBy

    def refuse(self):
        raise AssertionError("ghost_rearguard_coherence iterated a groupby")

    monkeypatch.setattr(DataFrameGroupBy, "__iter__", refuse)
    ghost_rearguard_coherence(_random_served(707))
