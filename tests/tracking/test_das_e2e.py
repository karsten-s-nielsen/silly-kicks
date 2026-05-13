"""DAS per-provider e2e tests (TF-28).

Full pipeline: load frames -> smooth_frames -> derive_velocities ->
infer_ball_carrier -> derive_team_in_possession -> get_das / das_at_action.

Uses tests/tracking/_provider_inputs.py loader + synthesize_actions for
consistent action synthesis across providers.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("accessible_space")

from silly_kicks.tracking import (
    derive_team_in_possession,
    infer_ball_carrier,
)
from silly_kicks.tracking._das import get_das
from silly_kicks.tracking.features import add_das, das_at_action
from silly_kicks.tracking.preprocess import derive_velocities, smooth_frames
from tests.tracking._provider_inputs import (
    GRADIENTSPORTS_DIR,
    SLIM_DIR,
    load_provider_frames,
    synthesize_actions,
)

pytestmark = pytest.mark.e2e

# Available providers: slim-parquet providers + gradientsports
_SLIM_PROVIDERS = sorted(p.stem.replace("_slim", "") for p in SLIM_DIR.glob("*_slim.parquet"))
_PROVIDERS = _SLIM_PROVIDERS + (["gradientsports"] if GRADIENTSPORTS_DIR.exists() else [])


def _prepare_provider(provider: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load frames, add velocities + team_in_possession, synthesize actions."""
    frames = load_provider_frames(provider)
    if "vx" not in frames.columns:
        frames = derive_velocities(smooth_frames(frames))
    carrier = infer_ball_carrier(frames)
    frames_with_poss = derive_team_in_possession(frames, carrier)
    actions = synthesize_actions(frames)
    return frames_with_poss, actions


@pytest.fixture(params=_PROVIDERS)
def provider_data(request) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    provider = request.param
    frames, actions = _prepare_provider(provider)
    return provider, frames, actions


class TestGetDasE2E:
    def test_output_has_as_das_columns(self, provider_data) -> None:
        provider, frames, _actions = provider_data
        result = get_das(frames, use_progress_bar=False)
        assert "AS" in result.columns, f"{provider}: missing AS column"
        assert "DAS" in result.columns, f"{provider}: missing DAS column"

    def test_output_length_matches_input(self, provider_data) -> None:
        provider, frames, _actions = provider_data
        result = get_das(frames, use_progress_bar=False)
        assert len(result) == len(frames), f"{provider}: output length {len(result)} != input {len(frames)}"

    def test_das_dtype_float64(self, provider_data) -> None:
        _provider, frames, _actions = provider_data
        result = get_das(frames, use_progress_bar=False)
        assert result["DAS"].dtype == np.float64

    def test_not_all_nan(self, provider_data) -> None:
        provider, frames, _actions = provider_data
        result = get_das(frames, use_progress_bar=False)
        valid = result["DAS"].dropna()
        assert len(valid) > 0, f"{provider}: all DAS values are NaN"


class TestDasAtActionE2E:
    def test_das_at_action_runs(self, provider_data) -> None:
        _provider, frames, actions = provider_data
        result = das_at_action(actions, frames)
        assert len(result) == len(actions)
        assert result.name == "das_team"
        assert result.dtype == np.float64

    def test_add_das_adds_three_columns(self, provider_data) -> None:
        provider, frames, actions = provider_data
        enriched = add_das(actions, frames)
        for col in ("das_team", "das_opponent", "das_diff"):
            assert col in enriched.columns, f"{provider}: missing {col}"
        assert len(enriched) == len(actions)
