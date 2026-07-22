"""TF-19 PR-3b seam test: build_ghost_frames -> provenance_to_targets -> xs_substitution_probe
runs end-to-end and is NON-VACUOUS (the ghost actually moves the keeper).

Does NOT re-test the probe's pass/fail logic (tests/tracking/test_probe_discriminating_power.py
covers that with hand-built targets). This closes the seam nobody else exercises (spec 1.2).
"""

from __future__ import annotations

from silly_kicks.gkdv import build_ghost_frames, provenance_to_targets
from silly_kicks.tracking._model_eval import (
    _TARGET_COLUMNS,
    _validate_targets,
    substitution_deltas,
    xs_substitution_probe,
)
from tests.gkdv._fixtures import multi_frame_in_domain
from tests.tracking._probe_fixtures import planted_model
from tests.tracking.test_ghost_gk import _fitted_model

_VERDICTS = {"pass", "fail", "unmeasurable_at_dose", "no_valid_placebo", "band_pass_flat_dose_response"}


def _chain(n_frames: int = 30):
    # C6: 30 frames -> ~6 scored (stride 5), so .any() isn't a 2-sample coin flip.
    # P1 verified multi_frame_in_domain already carries vx/vy -> no fixture augmentation needed.
    frames = multi_frame_in_domain(n_frames=n_frames)
    ghost_model = _fitted_model()[0]
    _cf, prov, report = build_ghost_frames(frames, model=ghost_model, home_team_id=1)
    targets = provenance_to_targets(prov, frames=frames, home_team_id=1)
    return frames, targets, report


def test_targets_match_probe_contract():
    _frames, targets, _report = _chain()
    assert list(targets.columns) == list(_TARGET_COLUMNS)
    _validate_targets(targets)  # RED if the adapter drifts from the probe's 7-col contract


def test_seam_runs_end_to_end():
    frames, targets, _report = _chain()
    out = xs_substitution_probe(planted_model("mixed"), frames, targets, seed=42)
    assert out["verdict"] in _VERDICTS  # RED if the probe surface changes under us
    assert out.get("n_frames_used", 0) >= 1  # RED if the chain silently produced no usable frames


def test_ghost_actually_moves_the_keeper_non_vacuity():
    # Load-bearing guard: if the ghost collapses onto the actual GK, the whole probe is a silent
    # null (spec 5, trap #1). Prove at least one scored frame has a non-zero keeper displacement.
    frames, targets, report = _chain()
    assert report.n_frames_scored > 0, "engine scored no frames — domain/fixture mismatch"
    deltas = substitution_deltas(planted_model("mixed"), frames, arm="xs", mode="targets", targets=targets, seed=42)
    gk = deltas[deltas["actor_role"] == "gk"]
    assert len(gk) > 0, "no gk-actor deltas produced"
    assert (gk["displacement_m"] > 0).any(), "ghost did not move the keeper — VACUOUS counterfactual"
