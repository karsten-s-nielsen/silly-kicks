"""compute_collective_variables contract + kernel-equivalence (TF-58 Task 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._collective import (
    COLLECTIVE_VARIABLES,
    collective_from_positions,
    compute_collective_variables,
    pack_groups,
)

_METRICS = list(COLLECTIVE_VARIABLES)


def _cv_frames(team_dtype: str = "int") -> pd.DataFrame:
    # 2 frames, 2 teams; each team 3 outfielders + 1 GK; plus a ball row and a NaN-position outfielder.
    rows = []
    for frame_id in (1, 2):
        for team_id in (10, 20):
            for x, y in [(0.0, 0.0), (4.0, 0.0), (2.0, 3.0 * frame_id)]:
                rows.append((frame_id, team_id, False, False, x, y))
            rows.append((frame_id, team_id, False, True, 1.0, 1.0))  # GK
            rows.append((frame_id, team_id, False, False, np.nan, np.nan))  # undetected outfielder
        rows.append((frame_id, 10, True, False, 50.0, 34.0))  # ball
    df = pd.DataFrame(rows, columns=["frame_id", "team_id", "is_ball", "is_goalkeeper", "x", "y"])
    df.insert(0, "period_id", 1)
    df.insert(0, "game_id", 1)
    if team_dtype == "str":
        df["team_id"] = df["team_id"].astype(str)
    elif team_dtype == "category":
        df["team_id"] = df["team_id"].astype("category")
    return df


def test_contract_columns_and_dtypes():
    out = compute_collective_variables(_cv_frames())
    assert list(out.columns) == ["game_id", "period_id", "frame_id", "team_id", "n_players", *_METRICS]
    assert out["n_players"].dtype == "Int64"
    for m in _METRICS:
        assert out[m].dtype == np.float64
    assert (out["n_players"] == 3).all()  # 3 valid outfielders per team-frame


def test_include_goalkeeper_changes_n_players():
    off = compute_collective_variables(_cv_frames(), include_goalkeeper=False)
    on = compute_collective_variables(_cv_frames(), include_goalkeeper=True)
    assert (off["n_players"] == 3).all()
    assert (on["n_players"] == 4).all()


def test_equals_kernel_on_packed_rows():
    frames = _cv_frames()
    out = compute_collective_variables(frames)
    mask = (~frames["is_ball"]) & (~frames["is_goalkeeper"]) & frames["x"].notna() & frames["y"].notna()
    outfield = frames[mask]
    gb = outfield.groupby(["game_id", "period_id", "frame_id", "team_id"], dropna=False, sort=True, observed=True)
    pos, counts, _ = pack_groups(
        gb.ngroup().to_numpy(),
        outfield["x"].to_numpy(dtype="float64"),
        outfield["y"].to_numpy(dtype="float64"),
        gb.ngroups,
    )
    cv = collective_from_positions(pos, counts)
    for m in _METRICS:
        np.testing.assert_array_equal(out[m].to_numpy(dtype="float64"), cv[m])


def test_ball_rows_and_nan_positions_excluded():
    frames = _cv_frames()
    # dropping the ball + NaN rows must not change any metric or count
    keep = ~(frames["is_ball"] | (frames["x"].isna()))
    trimmed = compute_collective_variables(frames[keep])
    full = compute_collective_variables(frames)
    pd.testing.assert_frame_equal(trimmed.reset_index(drop=True), full.reset_index(drop=True), check_exact=True)


@pytest.mark.parametrize("col", ["game_id", "team_id", "is_ball", "is_goalkeeper", "x", "y"])
def test_missing_required_column_raises(col):
    frames = _cv_frames().drop(columns=[col])
    with pytest.raises(ValueError, match=col):
        compute_collective_variables(frames)


def test_id_dtype_invariance():
    outs = {d: compute_collective_variables(_cv_frames(d)) for d in ("int", "str", "category")}
    for m in _METRICS:
        np.testing.assert_array_equal(outs["int"][m].to_numpy(float), outs["str"][m].to_numpy(float))
        np.testing.assert_array_equal(outs["int"][m].to_numpy(float), outs["category"][m].to_numpy(float))
    # team_id restored to the source dtype
    assert all(type(v) is str for v in outs["str"]["team_id"])


def test_does_not_mutate_input():
    frames = _cv_frames()
    before = frames.copy(deep=True)
    compute_collective_variables(frames)
    pd.testing.assert_frame_equal(frames, before, check_exact=True)
