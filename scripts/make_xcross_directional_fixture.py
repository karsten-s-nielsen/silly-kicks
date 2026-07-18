#!/usr/bin/env python
"""Build the frozen directional feature-vector fixture for the xCross CI liveness tripwire.

Companion to ``scripts/make_xshot_directional_fixture.py``. That script ranks real slim-frame
rows by ``r`` (distance to goal) -- a single, cleanly-rankable proxy for the xShot signal.
xCross has no such single proxy: "a cross is imminent" is a MULTI-feature condition (the ball is
wide AND advanced AND moving fast toward goal AND a teammate is arriving in the box), and the
model's dominant feature is ``ball_speed`` (permutation-importance #1), not a positional distance.
Real frames rarely separate into a clean cross-imminent-vs-not binary along any one feature, so the
tripwire is built from SYNTHETIC directional states with realistic, varying features instead. The
label is the constructed class; a live model must rank cross-imminent > not, a dead/constant model
cannot. (A y-invariance note: the discriminating geometry here is x-based -- distance to the
attacked endline plus ball speed -- so this fixture is orientation-robust; both flanks, y=8 and
y=60, are included to prove flank symmetry.)

Why this exists (regenerate-on-weights-bump): the previous committed fixture (frozen in 4.18.0,
PR-S85) held ``ball_speed == 0`` for every row -- it zeroed the model's #1 feature, so the model
correctly assigned ~0 to all rows and the pass/fail hinged on a razor-thin ordering of near-zero
predictions. That ordering flipped between the 4.18.0 weights and the TF-19 PR-2 y-correct retrain
(AUC 1.0 -> 0.25 on the SAME near-zero outputs), exposing the fixture as a degenerate probe rather
than a real tripwire. This generator gives the ball a realistic speed and varied geometry so the
gate exercises the model's actual learned signal. Validated to score AUC 1.0 on BOTH the 4.18.0 and
the PR-2 models -- i.e. it discriminates a live model from a dead one, which is the point.

Run once to (re)generate the committed parquet:
    python scripts/make_xcross_directional_fixture.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from silly_kicks.tracking._xcross_attempt import (
    XCROSS_FEATURE_NAMES_FAITHFUL,
    XCrossAttemptModel,
    extract_xcross_features,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "tests" / "datasets" / "tracking" / "xcross_directional" / "frozen_rows.parquet"

# Team 1 defends the x=0 goal (GK at (gk_x, gk_y)); team 2 (in possession) carries the ball. The
# ball's speed is set through its vx magnitude -- extract_xcross_features derives
# ``ball_speed = hypot(vx, vy)`` (magnitude only; ball direction is unused), so vx=-ball_speed
# (moving toward x=0) yields exactly ``ball_speed``.
_DEF_TEAM_ID = 1
_POSS_TEAM_ID = 2
_CARRIER_ID = 21


def _build_frame(fid: int, *, ball_x: float, ball_y: float, ball_speed: float, mate_near: bool) -> pd.DataFrame:
    def _r(pid, tid, x, y, *, is_ball=False, is_gk=False, vx=0.0, vy=0.0):
        return dict(
            player_id=pid, team_id=tid, is_ball=is_ball, is_goalkeeper=is_gk,
            x=x, y=y, frame_id=fid, time_seconds=0.0, vx=vx, vy=vy,
        )  # fmt: skip

    rows = [
        _r(-1, -1, ball_x, ball_y, is_ball=True, vx=-ball_speed, vy=0.0),  # ball toward x=0
        _r(10, _DEF_TEAM_ID, 2.0, 34.0, is_gk=True),  # defending GK
        _r(20, _POSS_TEAM_ID, 103.0, 34.0, is_gk=True),  # attacking GK
        _r(_CARRIER_ID, _POSS_TEAM_ID, ball_x + 0.3, ball_y),  # carrier (possession = team 2)
    ]
    # defenders (team 1) in/near the attacked box (small x)
    rows += [_r(11 + k, _DEF_TEAM_ID, 5.0 + 2 * k, 26.0 + 3 * k) for k in range(4)]
    # one teammate arriving near the carrier (mate_near) or far away; plus box attackers
    mate_dx, mate_dy = (3.0, 2.0) if mate_near else (28.0, -20.0)
    rows += [_r(30, _POSS_TEAM_ID, ball_x + mate_dx, ball_y + mate_dy)]
    rows += [_r(31 + k, _POSS_TEAM_ID, 9.0 + 2 * k, 30.0 + 2 * k) for k in range(3)]
    f = pd.DataFrame(rows)
    f["game_id"] = "g"
    f["period_id"] = 1
    f["z"] = 0.0
    f["frame_rate"] = 10.0
    f["ball_state"] = "alive"
    f["source_provider"] = "synthetic"
    return f


# Positives -- cross-imminent: wide (both flanks y=8 and y=60), advanced (small x), fast ball toward
# goal, teammate arriving in the box (mate_near=True, applied by label below).
_POS_SPECS: list[dict[str, float]] = [
    dict(ball_x=4.0, ball_y=8.0, ball_speed=9.0),
    dict(ball_x=6.0, ball_y=8.0, ball_speed=11.0),
    dict(ball_x=8.0, ball_y=60.0, ball_speed=8.0),
    dict(ball_x=10.0, ball_y=60.0, ball_speed=12.0),
    dict(ball_x=5.0, ball_y=10.0, ball_speed=10.0),
    dict(ball_x=7.0, ball_y=58.0, ball_speed=13.0),
    dict(ball_x=9.0, ball_y=9.0, ball_speed=7.5),
    dict(ball_x=11.0, ball_y=59.0, ball_speed=9.5),
]
# Negatives -- not a cross: central, deep (midfield), slow ball, teammate far (mate_near=False).
_NEG_SPECS: list[dict[str, float]] = [
    dict(ball_x=40.0, ball_y=34.0, ball_speed=1.5),
    dict(ball_x=45.0, ball_y=32.0, ball_speed=2.0),
    dict(ball_x=50.0, ball_y=36.0, ball_speed=1.0),
    dict(ball_x=42.0, ball_y=34.0, ball_speed=2.5),
    dict(ball_x=48.0, ball_y=30.0, ball_speed=1.2),
    dict(ball_x=52.0, ball_y=38.0, ball_speed=0.8),
    dict(ball_x=44.0, ball_y=33.0, ball_speed=1.8),
    dict(ball_x=46.0, ball_y=35.0, ball_speed=2.2),
]


def main() -> None:
    rows = []
    for label, specs, mate_near in ((1, _POS_SPECS, True), (0, _NEG_SPECS, False)):
        for fid, spec in enumerate(specs):
            frame = _build_frame(fid, mate_near=mate_near, **spec)
            feat = extract_xcross_features(
                frame, gk_team_id=_DEF_TEAM_ID, goal_x=0.0, carrier_player_id=_CARRIER_ID, score_differential=np.nan
            )
            rec = feat.iloc[0].to_dict()
            rec["label"] = label
            rows.append(rec)
    frozen = pd.DataFrame(rows)[[*XCROSS_FEATURE_NAMES_FAITHFUL, "label"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frozen.to_parquet(OUT)

    # Self-check: the bundled model must rank the directional fixture (a live model separates the
    # classes; a dead/constant one cannot). Printed, not asserted -- the CI gate lives in
    # tests/tracking/test_xcross_attempt_integration.py::test_xcross_bundled_model_is_live_not_degenerate.
    from sklearn.metrics import roc_auc_score

    model = XCrossAttemptModel.from_variant("default")
    p = model.predict_proba(frozen[XCROSS_FEATURE_NAMES_FAITHFUL])
    auc = roc_auc_score(frozen["label"].to_numpy(), p)
    labels = frozen["label"].to_numpy()
    pos_mean, neg_mean = p[labels == 1].mean(), p[labels == 0].mean()
    print(f"Wrote {len(frozen)} rows ({int(labels.sum())} cross-imminent/pos) to {OUT}")
    print(f"  bundled-model self-check AUC = {auc:.4f} (gate requires >= 0.9)")
    print(f"  pos mean p = {pos_mean:.5f}   neg mean p = {neg_mean:.5f}")


if __name__ == "__main__":
    main()
