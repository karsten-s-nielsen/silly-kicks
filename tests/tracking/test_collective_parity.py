"""Parity gates for the vectorised collective kernel (TF-58 D12/D13, spec 7.5, 9.2).

compute_defensive_line must stay byte-identical; compute_team_shape must stay byte-identical except
convex_hull_area (<= 1e-9 relative where Qhull succeeds; <= 1e-9 m^2 absolute where Qhull raised on a
precision-flat, not exactly collinear, frame -- owner ruling R3). The oracle is the frozen 05cfa56 copy.

At Task 1 production is unchanged, so new == legacy exactly and this file PASSES (proving the oracle is a
faithful copy). Task 3 makes production delegate to the kernel; this same file is then the regression gate.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import ConvexHull, QhullError

from silly_kicks.id_compat import ids_match
from silly_kicks.tracking import compute_defensive_line, compute_team_shape, resolve_defended_goals
from tests.tracking._legacy_collective_oracle import (
    legacy_compute_defensive_line,
    legacy_compute_team_shape,
)

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_FRAME_FIXTURES = (
    "tests/datasets/elastic_sync/j03wmx_slice/frames.parquet",
    "tests/datasets/tracking/action_context_slim/sportec_slim.parquet",
    "tests/datasets/tracking/action_context_slim/metrica_slim.parquet",
    "tests/datasets/tracking/action_context_slim/skillcorner_slim.parquet",
    "tests/datasets/sportec/idsse_slice/idsse_oldpath_harness_golden.parquet",
    "tests/datasets/tracking/synthetic/brief_outfielder.parquet",
    "tests/datasets/tracking/synthetic/gk_substitution.parquet",
    "tests/datasets/tracking/synthetic/sweeper_keeper.parquet",
)
_DL_VARIANTS = [(3, 5), (4, 5), (5, 5), ("adaptive", 3), ("adaptive", 4), ("adaptive", 5)]


def _frames(rel: str) -> pd.DataFrame:
    return pd.read_parquet(_ROOT / rel)


def _outcome(fn, *args, **kwargs):
    """(result, None) or (None, (exception type, message)) -- parity covers the raise path too."""
    try:
        return fn(*args, **kwargs), None
    except Exception as e:
        return None, (type(e), str(e))


def _team_ids(frames: pd.DataFrame) -> list:
    tid = frames.loc[~frames["is_ball"].astype(bool), "team_id"]
    return [t for t in pd.unique(tid) if pd.notna(t)]


def _team_pointsets(frames: pd.DataFrame, team_id) -> dict[tuple, np.ndarray]:
    mask = (
        ids_match(frames["team_id"], team_id)
        & (~frames["is_ball"].astype(bool))
        & (~frames["is_goalkeeper"].astype(bool))
        & frames["x"].notna()
        & frames["y"].notna()
    )
    out: dict[tuple, np.ndarray] = {}
    for key, g in frames[mask].groupby(["game_id", "period_id", "frame_id"], observed=True, dropna=False):
        out[key] = np.column_stack([g["x"].to_numpy(dtype="float64"), g["y"].to_numpy(dtype="float64")])
    return out


def _hull_class(pts: np.ndarray) -> tuple[str, float]:
    """(class, legacy_value): 'nan' (n<3), 'collinear' (exactly collinear -> 0.0), 'flat' (Qhull precision
    failure but not exactly collinear -> 0.0), or 'ok' (the ConvexHull area)."""
    if len(pts) < 3:
        return "nan", float("nan")
    try:
        return "ok", float(ConvexHull(pts).volume)
    except QhullError:
        rel = pts - pts[0]
        far = int(np.argmax(np.hypot(rel[:, 0], rel[:, 1])))
        vx, vy = rel[far]
        det = vx * rel[:, 1] - vy * rel[:, 0]
        return ("collinear", 0.0) if np.all(det == 0.0) else ("flat", 0.0)


@pytest.mark.parametrize("rel", _FRAME_FIXTURES)
@pytest.mark.parametrize(("n", "amn"), _DL_VARIANTS)
def test_defensive_line_byte_identical_to_legacy(rel, n, amn):
    frames = _frames(rel)
    goal_map = resolve_defended_goals(frames)
    new, new_err = _outcome(compute_defensive_line, frames, goal_map=goal_map, n=n, adaptive_max_n=amn)
    old, old_err = _outcome(legacy_compute_defensive_line, frames, goal_map=goal_map, n=n, adaptive_max_n=amn)
    assert new_err == old_err
    if old is not None:
        assert new is not None  # parity: same (result, err) shape, so new is not None when old is not
        pd.testing.assert_frame_equal(new, old, check_exact=True, check_dtype=True)


@pytest.mark.parametrize("rel", _FRAME_FIXTURES)
def test_team_shape_matches_legacy(rel):
    frames = _frames(rel)
    checked_ok = checked_zero = 0
    for tid in _team_ids(frames):
        new = compute_team_shape(frames, team_id=tid)
        old = legacy_compute_team_shape(frames, team_id=tid)
        # every column except convex_hull_area is byte-identical
        cols = [c for c in old.columns if c != "convex_hull_area"]
        pd.testing.assert_frame_equal(new[cols], old[cols], check_exact=True, check_dtype=True)
        # convex_hull_area: classify each row by recomputing the hull, then apply the R3 tolerance
        points = _team_pointsets(frames, tid)
        new_h = new["convex_hull_area"].to_numpy(dtype="float64")
        old_h = old["convex_hull_area"].to_numpy(dtype="float64")
        keys = list(zip(old["game_id"], old["period_id"], old["frame_id"], strict=True))
        for i, key in enumerate(keys):
            cls, legacy_val = _hull_class(points[key])
            if cls == "nan":
                assert np.isnan(new_h[i]) and np.isnan(old_h[i])
            elif cls == "collinear":
                assert new_h[i] == 0.0 and old_h[i] == 0.0
                checked_zero += 1
            elif cls == "flat":
                assert abs(new_h[i]) <= 1e-9 and old_h[i] == 0.0
                checked_zero += 1
            else:  # ok
                assert old_h[i] == pytest.approx(legacy_val, rel=1e-9, abs=0.0)
                assert new_h[i] == pytest.approx(old_h[i], rel=1e-9, abs=0.0)
                checked_ok += 1
    assert checked_ok > 0, f"{rel}: no successful-hull row exercised"


def _cut_tie_count(frames: pd.DataFrame) -> int:
    n = 0
    mask = (~frames["is_ball"].astype(bool)) & (~frames["is_goalkeeper"].astype(bool)) & frames["x"].notna()
    for _key, g in frames[mask].groupby(["game_id", "period_id", "frame_id", "team_id"], observed=True, dropna=False):
        xs = g["x"].to_numpy(dtype="float64")
        asc = np.sort(xs)
        desc = -np.sort(-xs)
        for k in (3, 4, 5):
            if len(xs) > k and (asc[k - 1] == asc[k] or desc[k - 1] == desc[k]):
                n += 1
                break
    return n


def test_parity_fixtures_contain_cut_ties():
    """ADR-032 precondition: the default-argsort tie path (C1) is actually exercised."""
    j03 = _frames("tests/datasets/elastic_sync/j03wmx_slice/frames.parquet")
    assert _cut_tie_count(j03) >= 50


def test_parity_fixtures_contain_rtl_and_adaptive_groups():
    """ADR-032 precondition: rtl teams and >=6-outfield groups (the adaptive n=5 cut) appear in the corpus
    fixtures. The sub-3-outfield parity path is covered separately by the synthetic case below, because no
    committed fixture happens to carry a sub-3 frame-team."""
    has_rtl = has_adaptive = False
    for rel in _FRAME_FIXTURES:
        frames = _frames(rel)
        if "team_attacking_direction" in frames.columns:
            has_rtl = has_rtl or "rtl" in set(frames["team_attacking_direction"].dropna().unique())
        mask = (~frames["is_ball"].astype(bool)) & (~frames["is_goalkeeper"].astype(bool)) & frames["x"].notna()
        sizes = (
            frames[mask].groupby(["game_id", "period_id", "frame_id", "team_id"], observed=True, dropna=False).size()
        )
        if (sizes >= 6).any():
            has_adaptive = True
    assert has_rtl, "no rtl team in the fixture union"
    assert has_adaptive, "no >=6-outfield group in the fixture union"


def _sub3_frames() -> pd.DataFrame:
    """One frame: team 1 (ltr, GK + 2 outfielders -> n<3), team 2 (rtl, GK + 4 outfielders)."""
    rows = [
        (1, True, 4.0, 34.0, "ltr"),
        (1, False, 20.0, 30.0, "ltr"),
        (1, False, 30.0, 40.0, "ltr"),
        (2, True, 101.0, 34.0, "rtl"),
        (2, False, 80.0, 20.0, "rtl"),
        (2, False, 70.0, 30.0, "rtl"),
        (2, False, 60.0, 40.0, "rtl"),
        (2, False, 50.0, 50.0, "rtl"),
    ]
    return pd.DataFrame(
        {
            "game_id": 1,
            "period_id": 1,
            "frame_id": 1,
            "player_id": list(range(len(rows))),
            "team_id": [r[0] for r in rows],
            "is_ball": False,
            "is_goalkeeper": [r[1] for r in rows],
            "x": [r[2] for r in rows],
            "y": [r[3] for r in rows],
            "team_attacking_direction": [r[4] for r in rows],
        }
    )


def test_parity_on_synthetic_sub3_group():
    """ADR-032 non-vacuity for the <3-outfield path: both functions match legacy byte-for-byte when a
    frame-team has fewer than 3 outfielders (compute_team_shape -> hull NaN; compute_defensive_line -> NaN row)."""
    frames = _sub3_frames()
    goal_map = resolve_defended_goals(frames)
    for tid in (1, 2):
        pd.testing.assert_frame_equal(
            compute_team_shape(frames, team_id=tid),
            legacy_compute_team_shape(frames, team_id=tid),
            check_exact=True,
            check_dtype=True,
        )
    pd.testing.assert_frame_equal(
        compute_defensive_line(frames, goal_map=goal_map, n=4),
        legacy_compute_defensive_line(frames, goal_map=goal_map, n=4),
        check_exact=True,
        check_dtype=True,
    )
    # the <3 branch is really hit: team 1 has 2 outfielders -> its team-shape hull is NaN
    ts1 = compute_team_shape(frames, team_id=1)
    assert bool(ts1["convex_hull_area"].isna().all())
