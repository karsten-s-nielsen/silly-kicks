"""Bit-exact per-row regression gate for add_action_context + add_pre_shot_gk_position
(PR-S21 TF-11).

Per ADR-005 hybrid validation: per-row gate is load-bearing; JSON baselines are
documentation. This file is the per-row gate.

Failure mode: ``row 5: pre_shot_gk_distance_to_shot expected 4.32, got 4.51`` —
fully debuggable. No statistical noise tolerance.

Inputs are loaded via the shared module ``tests/tracking/_provider_inputs.py``,
which is also imported by ``scripts/regenerate_action_context_baselines.py`` —
keeps regeneration and CI assertion in sync.

The slim slices contain non-shot synthesized actions (per PR-S20 cross-provider test
pattern). Hence GK-position columns flow through as all-NaN — the gate verifies the
all-NaN regression path. Real GK-position behavior is validated in:
  - Tier-1 analytical kernel tests (tests/tracking/test_kernels.py)
  - e2e real-data sweep (tests/tracking/test_action_context_real_data_sweep.py)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.spadl.utils import add_pre_shot_gk_context
from silly_kicks.tracking.features import add_action_context, add_pre_shot_gk_position
from tests.tracking._provider_inputs import (
    N_ACTIONS_PER_PROVIDER,
    load_provider_frames,
    synthesize_actions,
)

SLIM_DIR = Path(__file__).resolve().parent.parent / "datasets" / "tracking" / "action_context_slim"

EXPECTED_COLUMNS = [
    "action_id",
    "nearest_defender_distance",
    "actor_speed",
    "receiver_zone_density",
    "defenders_in_triangle_to_goal",
    "pre_shot_gk_x",
    "pre_shot_gk_y",
    "pre_shot_gk_distance_to_goal",
    "pre_shot_gk_distance_to_shot",
    "frame_id",
    "time_offset_seconds",
    "link_quality_score",
    "n_candidate_frames",
]


@pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner", "pff"])
def test_add_action_context_per_row_regression(provider: str) -> None:
    expected_path = SLIM_DIR / f"{provider}_expected.parquet"
    if not expected_path.exists():
        pytest.skip(f"{expected_path} not committed; run scripts/regenerate_action_context_baselines.py.")
    frames = load_provider_frames(provider)
    actions = synthesize_actions(frames, n_actions=N_ACTIONS_PER_PROVIDER)

    actions = add_pre_shot_gk_context(actions)
    out_ac = add_action_context(actions, frames)
    out_gk = add_pre_shot_gk_position(actions, frames)
    actual = out_ac.copy()
    for col in (
        "pre_shot_gk_x",
        "pre_shot_gk_y",
        "pre_shot_gk_distance_to_goal",
        "pre_shot_gk_distance_to_shot",
    ):
        actual[col] = out_gk[col].values
    actual = actual[EXPECTED_COLUMNS].reset_index(drop=True)

    expected = pd.read_parquet(expected_path).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, atol=1e-9, rtol=0, check_dtype=True)


@pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner", "pff"])
def test_expected_parquet_has_real_gk_columns(provider: str) -> None:
    """Per-row regression gate must exercise REAL GK-position computation, not just the
    all-NaN path. Synthesizer (tests/tracking/_provider_inputs.py) stamps a synthetic
    keeper_save + shot at the chronological tail so add_pre_shot_gk_context populates
    defending_gk_player_id, and add_pre_shot_gk_position emits non-NaN GK position for
    the linked shot frame.

    Without this assertion, a bug that drops GK columns to all-NaN would silently
    regress to passing.
    """
    expected_path = SLIM_DIR / f"{provider}_expected.parquet"
    if not expected_path.exists():
        pytest.skip(f"{expected_path} not committed.")
    df = pd.read_parquet(expected_path)
    n_valid_gk_x = df["pre_shot_gk_x"].notna().sum()
    assert n_valid_gk_x >= 1, (
        f"{provider}: pre_shot_gk_x has {n_valid_gk_x} non-NaN rows; expected >=1 from "
        f"the synthesized shot. Check tests/tracking/_provider_inputs.py::synthesize_actions."
    )
    # Where pre_shot_gk_x is non-NaN, the other 3 GK columns must also be non-NaN
    # (kernel writes them together).
    valid_mask = df["pre_shot_gk_x"].notna()
    for col in (
        "pre_shot_gk_y",
        "pre_shot_gk_distance_to_goal",
        "pre_shot_gk_distance_to_shot",
    ):
        n_valid = df.loc[valid_mask, col].notna().sum()
        assert n_valid == n_valid_gk_x, (
            f"{provider}: {col} has {n_valid} non-NaN rows where pre_shot_gk_x has {n_valid_gk_x}"
        )
