"""Cross-provider parity gate --- schema-stress test with all 4 providers.

Per spec § 5.5 / ADR-004 invariant 2: the canonical schema is locked only
after all four providers' adapters produce the same shape. This file
runs the parity assertions against committed synthetic fixtures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import kloppy as kloppy_gw
from silly_kicks.tracking import pff, sportec
from silly_kicks.tracking.schema import TRACKING_CONSTRAINTS
from silly_kicks.tracking.utils import link_actions_to_frames

_FIX = Path(__file__).resolve().parent / "datasets" / "tracking"


def _load_sportec():
    raw = pd.read_parquet(_FIX / "sportec" / "medium_halftime.parquet")
    return sportec.convert_to_frames(
        raw,
        home_team_id="DFL-CLU-0100",
        home_team_start_left=True,
    )


def _load_pff():
    raw = pd.read_parquet(_FIX / "pff" / "medium_halftime.parquet")
    return pff.convert_to_frames(raw, home_team_id=100, home_team_start_left=True)


def _load_metrica():
    from datasets.tracking.metrica.generate_synthetic import (
        build_metrica_tracking_dataset,
    )

    ds = build_metrica_tracking_dataset(n_frames=750)
    return kloppy_gw.convert_to_frames(ds)


def _load_skillcorner():
    from datasets.tracking.skillcorner.generate_synthetic import (
        build_skillcorner_tracking_dataset,
    )

    ds = build_skillcorner_tracking_dataset(n_frames=300)
    return kloppy_gw.convert_to_frames(ds)


PROVIDER_LOADERS = {
    "sportec": _load_sportec,
    "pff": _load_pff,
    "metrica": _load_metrica,
    "skillcorner": _load_skillcorner,
}


@pytest.mark.parametrize("provider", list(PROVIDER_LOADERS))
def test_tracking_bounds(provider):
    frames, _ = PROVIDER_LOADERS[provider]()
    for col, (lo, hi) in TRACKING_CONSTRAINTS.items():
        if col not in frames.columns:
            continue
        vals = frames[col].dropna()
        if len(vals) == 0:
            continue
        if hi == float("inf"):
            assert (vals >= lo).all(), f"{provider}/{col}: values below {lo}"
        else:
            assert vals.between(lo, hi).all(), f"{provider}/{col}: values outside [{lo}, {hi}]"


@pytest.mark.parametrize("provider", list(PROVIDER_LOADERS))
def test_tracking_link_rate(provider):
    frames, _ = PROVIDER_LOADERS[provider]()
    sample = frames.drop_duplicates(["period_id", "frame_id"]).sample(20, random_state=42)
    actions = pd.DataFrame(
        {
            "game_id": 1,
            "action_id": range(20),
            "period_id": sample["period_id"].to_numpy(),
            "time_seconds": sample["time_seconds"].to_numpy(),
            "team_id": 100,
            "player_id": 7,
            "type_id": 0,
            "result_id": 1,
            "bodypart_id": 0,
            "start_x": 50.0,
            "start_y": 34.0,
            "end_x": 60.0,
            "end_y": 34.0,
        }
    )
    _pointers, report = link_actions_to_frames(actions, frames, tolerance_seconds=0.1)
    assert report.link_rate >= 0.95, f"{provider}: link rate {report.link_rate}"


def test_tracking_distance_to_ball_distribution_overlap():
    """Per-provider distance-to-ball percentiles overlap, proving coordinate
    normalization is consistent across providers (the schema-stress goal)."""
    percentiles = {}
    for prov, loader in PROVIDER_LOADERS.items():
        frames, _ = loader()
        ball = frames[frames["is_ball"]][["period_id", "frame_id", "x", "y"]].rename(
            columns={"x": "bx", "y": "by"},
        )
        players = frames[~frames["is_ball"]]
        joined = players.merge(ball, on=["period_id", "frame_id"])
        dist = np.sqrt((joined["x"] - joined["bx"]) ** 2 + (joined["y"] - joined["by"]) ** 2)
        percentiles[prov] = np.percentile(dist.dropna(), [25, 50, 75, 95]).tolist()
    for prov, p in percentiles.items():
        assert 0 < p[1] < 100, f"{prov}: implausible p50 distance {p[1]}"
    p50s = [p[1] for p in percentiles.values()]
    assert max(p50s) / min(p50s) < 3.0, f"p50 distance ratios: {percentiles}"
