"""Regenerate slim-parquet expected outputs + JSON distribution baselines (PR-S21 TF-11).

For each provider:
  1. Load committed slim parquet (Tier-3 lakehouse-derived for sportec/metrica/skillcorner;
     synthetic medium_halftime for pff). Frames only — actions in slim are placeholder rows
     with NULL identifiers per the lakehouse fct_action_values asymmetry. Mirrors the PR-S20
     _load_frames + _synthesize_actions pattern in tests/tracking/test_action_context_cross_provider.py.
  2. Synthesize 10 actions anchored on real (period_id, frame_id, player_id) triples from the
     committed frame slice; assign type_id=pass_id (synthesized actions are non-shots).
  3. Run silly_kicks.spadl.utils.add_pre_shot_gk_context(actions) to populate
     defending_gk_player_id (events-only step; produces NaN for non-shots).
  4. Run add_action_context(actions, frames) → 4 PR-S20 features + 4 provenance columns.
  5. Run add_pre_shot_gk_position(actions, frames) → 4 GK-position columns (all NaN since
     synthesized actions are non-shots; verifies the all-NaN regression path).
  6. Project to expected schema and write {provider}_expected.parquet.
  7. Compute p25/p50/p75/p99 per feature (per provider); populate JSON null slots.
     Slots remain null for features with all-NaN data (documented coverage gap; the JSON-shape
     test allows null only when the corresponding *_expected.parquet column is all-NaN).

Run: uv run python scripts/regenerate_action_context_baselines.py

Idempotent. Not part of CI. Reviewer-visible parquet diff via git diff.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from silly_kicks.spadl.utils import add_pre_shot_gk_context
from silly_kicks.tracking.features import add_action_context, add_pre_shot_gk_position
from tests.tracking._provider_inputs import (
    N_ACTIONS_PER_PROVIDER,
)
from tests.tracking._provider_inputs import (
    load_provider_frames as _load_provider_frames,
)
from tests.tracking._provider_inputs import (
    synthesize_actions as _synthesize_actions,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SLIM_DIR = REPO_ROOT / "tests" / "datasets" / "tracking" / "action_context_slim"
JSON_PATH = REPO_ROOT / "tests" / "datasets" / "tracking" / "empirical_action_context_baselines.json"

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
FEATURE_COLS_FOR_BASELINES = [
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


def _compute_expected(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Run events-only step + both aggregators; merge into expected-schema DataFrame."""
    actions = add_pre_shot_gk_context(actions)
    out_ac = add_action_context(actions, frames)
    out_gk = add_pre_shot_gk_position(actions, frames)
    out = out_ac.copy()
    for col in (
        "pre_shot_gk_x",
        "pre_shot_gk_y",
        "pre_shot_gk_distance_to_goal",
        "pre_shot_gk_distance_to_shot",
    ):
        out[col] = out_gk[col].values
    return out[EXPECTED_COLUMNS]


def _compute_percentiles(expected: pd.DataFrame) -> dict:
    out: dict[str, float | None] = {}
    for col in FEATURE_COLS_FOR_BASELINES:
        series = expected[col].dropna()
        for p in PERCENTILES:
            key = f"{col}_p{p}"
            if len(series) == 0:
                out[key] = None
            else:
                out[key] = float(np.percentile(series, p))
    return out


def main() -> None:
    json_state = json.loads(JSON_PATH.read_text())
    for provider in ("sportec", "metrica", "skillcorner", "pff"):
        frames = _load_provider_frames(provider)
        actions = _synthesize_actions(frames, n_actions=N_ACTIONS_PER_PROVIDER)
        expected = _compute_expected(actions, frames)
        expected_path = SLIM_DIR / f"{provider}_expected.parquet"
        expected.to_parquet(expected_path, index=False)
        print(f"{provider}: wrote {len(expected)} rows to {expected_path.name}")

        percentiles = _compute_percentiles(expected)
        for k, v in percentiles.items():
            json_state["providers"][provider][k] = v

    JSON_PATH.write_text(json.dumps(json_state, indent=2))
    print(f"Wrote updated baselines JSON: {JSON_PATH.name}")


if __name__ == "__main__":
    main()
