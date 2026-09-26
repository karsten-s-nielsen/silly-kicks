"""F1b (ADR-106) commit-1 gate: un-retrained bundled models load + finite-predict on float32 frames.

F1B-SPEC-01: commit-1 stores float32 frames but does NOT retrain (retrain is commit-2). The probe
builders (`canonical_probe_frame` / feature-contract probe) are schema-INDEPENDENT and stay float64 in
commit-1, so each un-retrained (float64) model's stored float64 fingerprint still matches at ``load()``
-- chirality + feature-contract pass. This gate proves exactly that, and that a bundled model then
finite-predicts on a float32-stored frame. If a probe builder were wrongly routed through the float32
schema in commit-1, that model's ``load()`` would go red here (the F1B-SPEC-01 trap).

After commit-2 retrains + casts the probe builders to float32, this gate keeps passing (float32 probe
matches float32 fingerprint), so it is durable across both commits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import silly_kicks.tracking as T

_FRAME_GEOMETRY_MODELS = [
    "XShotOccurrenceModel",
    "XCrossAttemptModel",
    "GhostGkModel",
    "GhostOutfieldModel",
    "ReceiverModel",
    "GkCompletionModel",
]


@pytest.mark.parametrize("model_name", _FRAME_GEOMETRY_MODELS)
def test_bundled_model_loads_on_unchanged_float64_probe(model_name):
    # load() runs verify_chirality + the feature-contract probe against the model's stored (float64,
    # commit-1-unchanged) probe. A raise here means a probe builder was routed through the float32
    # schema -- fix the builder to stay float64 until commit-2 casts it explicitly.
    cls = getattr(T, model_name)
    model = cls.from_variant("default")
    assert model is not None


def _float32_canonical_frame() -> pd.DataFrame:
    """One canonical-schema frame with float32 coords + category team_id (the F1b storage dtypes)."""
    rows = [
        dict(player_id=-1, team_id=-1, is_ball=True, is_goalkeeper=False, x=20.0, y=34.0),
        dict(player_id=10, team_id=1, is_ball=False, is_goalkeeper=True, x=2.0, y=34.0),
        dict(player_id=11, team_id=1, is_ball=False, is_goalkeeper=False, x=10.0, y=30.0),
        dict(player_id=12, team_id=1, is_ball=False, is_goalkeeper=False, x=12.0, y=38.0),
        dict(player_id=20, team_id=2, is_ball=False, is_goalkeeper=True, x=103.0, y=34.0),
        dict(player_id=21, team_id=2, is_ball=False, is_goalkeeper=False, x=20.3, y=34.0),
        dict(player_id=22, team_id=2, is_ball=False, is_goalkeeper=False, x=25.0, y=30.0),
    ]
    df = pd.DataFrame(rows)
    df["game_id"] = 1
    df["period_id"] = 1
    df["frame_id"] = 100
    df["time_seconds"] = 0.0
    df["frame_rate"] = 25.0
    for c in ("x", "y"):
        df[c] = df[c].astype("float32")
    df["z"] = np.float32(0.0)
    df["vx"] = np.float32(0.0)
    df["vy"] = np.float32(0.0)
    df["speed"] = np.float32(0.0)
    df["ball_state"] = "alive"
    df["player_id"] = df["player_id"].astype("Int64")
    df["team_id"] = df["team_id"].astype("Int64").astype("category")  # F1b storage dtype
    return df


def test_bundled_xshot_finite_predicts_on_float32_frame():
    # A representative frame-geometry model serves finite output on a float32-stored frame (the other
    # models' float32-serve is covered by the model real-data suites).
    model = T.XShotOccurrenceModel.from_variant("default")
    frame = _float32_canonical_frame()
    out = T.compute_xshot_occurrence(frame, model=model, home_team_id=1)
    vals = out["xshot_occurrence"].dropna()
    assert len(vals) >= 1
    assert np.all(np.isfinite(vals.to_numpy(dtype="float64")))
    assert vals.between(0.0, 1.0).all()
