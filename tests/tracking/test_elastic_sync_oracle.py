"""CI regression gate for ELASTIC-NW sync accuracy on the committed CC-BY oracle slice.

Scores ``align_events_to_frames`` against Kim et al.'s frame-level ground truth
(``tests/datasets/elastic_sync/j03wmx_slice/``; CC BY 4.0, see that dir's README) with a
from-scratch ``|pred - gt| <= N`` accuracy metric -- NEVER ELASTIC's MPL-2.0 ``compute_sync_accuracy``.

This is a REGRESSION FLOOR, not the headline. The floors are the achieved-on-this-slice accuracy
minus a safety margin (measured 2026-09-14 on this exact committed slice -- a J03WMX period-1 span
that COVERS a goal (t~468 s) -- with the central-difference accel + in-play episode grouping + the
evolve-tuned scoring of ADR-093; deterministic):

    START      exact 0.493  within-2 0.791  within-5 0.910  within-25 0.955  (n=67)
    RECEPTION  exact 0.471  within-2 0.784  within-5 0.922  within-25 0.941  (populated 63)
    do-nothing (nominal time*25) exact 0.000

The paper (arXiv:2608.30227) adopts **W2** (within-2-frames, 0.08 s) as its PRIMARY metric and reports
NO exact-frame headline (Table 2: ELASTIC-NW W2 96.5%, ELASTIC-Greedy W2 84.1%, on the paper's own
same-source data). Our production benchmark (the Claude-run DGX 3-match report, spec section 16) is
**W2 = 0.862 cross-source** -- ABOVE the paper's greedy (0.841) and ~10 pts below the paper's
same-source NW (0.965). That ~10-pt gap is the CROSS-SOURCE event-time misalignment, NOT the alignment
algorithm or player identity: this oracle scores Kim's re-annotated event labels against our DFL-OBJ
tracking anchored via a per-period vote-map, which leaves ~0.8 s of per-event event-vs-tracking jitter
(median anchor error ~20 frames, uncorrelated across neighbours). An OpenEvolve LLM code-search over
the scoring (ADR-093) converged early -- confirming the scoring is near its clean-room ceiling on this
data; ball height, velocity-direction, a time-proximity prior and a local time re-anchor were each
measured weak or dead. On same-source data the jitter is smaller, so production W2 should approach the
paper's same-source figure.

The committed slice is small and CI-runnable -> this is a REGULAR-suite test (not @e2e).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.id_compat import canonical_id
from silly_kicks.tracking import align_events_to_frames
from silly_kicks.tracking._elastic_sync import (
    ElasticSyncParams,
    _build_frame_lookups,
    _detect_candidate_frames,
    _enrich_events,
    _score,
)

_SLICE = "tests/datasets/elastic_sync/j03wmx_slice"

# achieved-minus-margin floors (see module docstring). Margins (~0.07-0.10) tolerate cross-platform
# float variation while still catching a real regression (e.g. reverting the central-diff accel drops
# start exact to ~0.335, below 0.42). Reception w5 achieved 0.922 -> floor 0.82.
_START_EXACT_FLOOR = 0.42
_START_W2_FLOOR = 0.70
_RECEIVE_W5_FLOOR = 0.82


def _load():
    a = pd.read_parquet(f"{_SLICE}/actions.parquet")
    f = pd.read_parquet(f"{_SLICE}/frames.parquet")
    g = pd.read_parquet(f"{_SLICE}/gt.parquet")
    return a, f, g


def sync_accuracy(pred_frame: pd.Series, gt_frame: pd.Series, *, tol: int) -> float:
    """Fraction of aligned events whose predicted frame is within ``tol`` of the ground-truth
    frame. Reimplemented from scratch (an ``|pred - gt| <= N`` count) -- not ELASTIC's metric."""
    d = (pd.to_numeric(pred_frame, errors="coerce") - pd.to_numeric(gt_frame, errors="coerce")).abs().dropna()
    return float((d <= tol).mean()) if len(d) else float("nan")


@pytest.fixture(scope="module")
def scored():
    actions, frames, gt = _load()
    pred = align_events_to_frames(actions, frames)
    m = pred.merge(gt[["action_id", "frame_id", "receive_frame_id"]], on="action_id", how="right")
    return m


def test_nw_clears_start_regression_floor(scored):
    exact = sync_accuracy(scored["elastic_frame_id"], scored["frame_id"], tol=0)
    w2 = sync_accuracy(scored["elastic_frame_id"], scored["frame_id"], tol=2)
    assert exact >= _START_EXACT_FLOOR, f"start exact-frame {exact:.3f} < floor {_START_EXACT_FLOOR}"
    assert w2 >= _START_W2_FLOOR, f"start within-2 {w2:.3f} < floor {_START_W2_FLOOR}"


def test_nw_reception_frames_populated_and_accurate(scored):
    populated = scored["elastic_receive_frame_id"].notna().sum()
    assert populated >= 10, f"only {populated} reception frames populated -- reception path inert"
    w5 = sync_accuracy(scored["elastic_receive_frame_id"], scored["receive_frame_id"], tol=5)
    assert w5 >= _RECEIVE_W5_FLOOR, f"reception within-5 {w5:.3f} < floor {_RECEIVE_W5_FLOOR}"


def test_nw_beats_greedy_argmax_baseline(scored):
    """NW must beat the ELASTIC-Greedy baseline by a LARGE margin (plan Task 17.2 -- replaces the
    trivial do-nothing baseline). Greedy = per-event argmax of the SAME ``_score``, WITHOUT the NW
    DP's order constraint; it is the constant-scorer-catching guard, because a broken/constant scorer
    collapses NW and greedy together, so a large NW-over-greedy margin proves BOTH that the scorer
    discriminates AND that the DP order-constraint is load-bearing. Measured on this slice: NW W2
    0.791 vs greedy 0.224 (+0.567); NW exact 0.493 vs greedy 0.194 (+0.299). A trivial do-nothing
    baseline (nominal clock, no candidate snapping) is asserted ~0% exact as the floor of the range."""
    actions, frames, gt = _load()
    params = ElasticSyncParams()
    cands = _detect_candidate_frames(frames, params=params)
    lookups = _build_frame_lookups(frames, params=params)
    events = _enrich_events(actions, params=params)
    real = {int(e.action_id): e for e in events if e.kind == "real" and pd.notna(e.action_id)}
    gp = (canonical_id(actions["game_id"].iloc[0]), int(actions["period_id"].iloc[0]))
    by_frame = {c.frame_id: c for c in cands[gp]}
    lk = lookups[gp]
    greedy = pd.DataFrame(
        [
            {
                "action_id": aid,
                "greedy": int(max(by_frame.values(), key=lambda c: _score(e, c, lk, params=params)).frame_id),
            }
            for aid, e in real.items()
        ]
    ).merge(gt[["action_id", "frame_id"]], on="action_id")
    g_exact = sync_accuracy(greedy["greedy"], greedy["frame_id"], tol=0)
    g_w2 = sync_accuracy(greedy["greedy"], greedy["frame_id"], tol=2)
    nw_exact = sync_accuracy(scored["elastic_frame_id"], scored["frame_id"], tol=0)
    nw_w2 = sync_accuracy(scored["elastic_frame_id"], scored["frame_id"], tol=2)
    assert nw_w2 >= g_w2 + 0.30, f"NW W2 {nw_w2:.3f} does not beat greedy {g_w2:.3f} by >=0.30"
    assert nw_exact >= g_exact + 0.15, f"NW exact {nw_exact:.3f} does not beat greedy {g_exact:.3f} by >=0.15"
    # trivial do-nothing baseline (nominal clock, no candidate snapping) is ~0% exact on this slice.
    base = {
        int(p): int(frames[(frames.period_id == p) & frames.is_ball]["frame_id"].min())
        for p in frames["period_id"].unique()
    }
    nominal = actions.assign(
        nom=[base[int(p)] + 25 * t for p, t in zip(actions["period_id"], actions["time_seconds"], strict=True)]
    ).merge(gt[["action_id", "frame_id"]], on="action_id")
    assert float((nominal["nom"].round() == nominal["frame_id"]).mean()) <= 0.05


def test_a_shuffled_alignment_fails_the_floor(scored):
    """Non-vacuity / both-sided: a mutation that scrambles the alignment must drop exact-frame
    below the floor -- proving the gate has teeth, not that any output passes."""
    rng = np.random.RandomState(0)
    shuffled = scored["elastic_frame_id"].to_numpy(dtype=float).copy()
    rng.shuffle(shuffled)
    exact = sync_accuracy(pd.Series(shuffled), scored["frame_id"], tol=0)
    assert exact < _START_EXACT_FLOOR, f"shuffled alignment still scored {exact:.3f} -- gate lacks teeth"
