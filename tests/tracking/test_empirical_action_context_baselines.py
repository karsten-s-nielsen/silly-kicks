"""JSON-shape gate for empirical_action_context_baselines.json (PR-S21 TF-11).

Promoted from advisory (PR-S20) to load-bearing CI gate. Two assertions:

1. Schema completeness: every (feature x percentile x provider) slot has a value
   (float OR null). Null is allowed only when the corresponding *_expected.parquet
   column is all-NaN — documented coverage gap (synthesized actions are non-shots,
   so GK-position percentiles are null until full real-data e2e captures shot rows).

2. JSON ↔ parquet consistency: percentiles in JSON match those re-computed from the
   *_expected.parquet (strict tolerance — JSON-vs-parquet check, not feature-vs-baseline).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets" / "tracking"
JSON_PATH = DATASETS_DIR / "empirical_action_context_baselines.json"
SLIM_DIR = DATASETS_DIR / "action_context_slim"

FEATURE_COLS = [
    "nearest_defender_distance",
    "actor_speed",
    "receiver_zone_density",
    "defenders_in_triangle_to_goal",
    "pre_shot_gk_x",
    "pre_shot_gk_y",
    "pre_shot_gk_distance_to_goal",
    "pre_shot_gk_distance_to_shot",
]
PERCENTILES = [25, 50, 75, 99]
PROVIDERS = ["sportec", "metrica", "skillcorner", "pff"]


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("feature", FEATURE_COLS)
@pytest.mark.parametrize("percentile", PERCENTILES)
def test_baselines_json_slot_present(provider: str, feature: str, percentile: int) -> None:
    """Every (provider x feature x percentile) slot exists in the JSON."""
    state = json.loads(JSON_PATH.read_text())
    key = f"{feature}_p{percentile}"
    assert provider in state["providers"], f"Missing provider entry: {provider}"
    assert key in state["providers"][provider], f"Missing key {provider}/{key}"


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("feature", FEATURE_COLS)
@pytest.mark.parametrize("percentile", PERCENTILES)
def test_baselines_json_null_only_when_expected_parquet_all_nan(
    provider: str,
    feature: str,
    percentile: int,
) -> None:
    """Null slots are documented coverage gaps: allowed only when the *_expected.parquet
    column is all-NaN for that feature."""
    state = json.loads(JSON_PATH.read_text())
    key = f"{feature}_p{percentile}"
    value = state["providers"][provider][key]
    if value is not None:
        return  # populated -> nothing to check here

    expected_path = SLIM_DIR / f"{provider}_expected.parquet"
    if not expected_path.exists():
        pytest.skip(f"{expected_path} not committed.")
    expected = pd.read_parquet(expected_path)
    assert expected[feature].isna().all(), (
        f"{provider}/{key}: JSON has null but {provider}_expected.parquet[{feature}] has non-NaN values."
    )


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("feature", FEATURE_COLS)
@pytest.mark.parametrize("percentile", PERCENTILES)
def test_baselines_json_matches_expected_parquet_distribution(
    provider: str,
    feature: str,
    percentile: int,
) -> None:
    """Sanity: percentile in JSON matches re-computed value from *_expected.parquet
    within strict tolerance. Catches accidental JSON hand-edits that drift from the data."""
    state = json.loads(JSON_PATH.read_text())
    key = f"{feature}_p{percentile}"
    json_value = state["providers"][provider][key]

    expected_path = SLIM_DIR / f"{provider}_expected.parquet"
    if not expected_path.exists():
        pytest.skip(f"{expected_path} not committed.")
    expected = pd.read_parquet(expected_path)
    series = expected[feature].dropna()

    if len(series) == 0:
        assert json_value is None, f"{provider}/{key}: empty data but JSON has non-null ({json_value})."
        return

    computed = float(np.percentile(series, percentile))
    assert json_value is not None, f"{provider}/{key}: JSON null but data has values."
    assert json_value == pytest.approx(computed, abs=1e-6, rel=0), (
        f"{provider}/{key}: JSON={json_value}, parquet-computed={computed}"
    )
