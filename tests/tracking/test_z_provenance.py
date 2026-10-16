"""ADR-125: validate_z_provenance cross-checks a declared z_source against the ball-z signature.

Flags a "measured" declaration with a PERSISTENT-ELEVATION (extrapolation) or flat/absent signature;
ground-level negatives (Gradient Sports) must NOT flag. Thresholds pre-registered against the spike.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from silly_kicks.tracking import ZProvenanceWarning, validate_z_provenance


def _ball_frames(z_values, z_source):
    n = len(z_values)
    return pd.DataFrame(
        {
            "game_id": [1] * n,
            "period_id": [1] * n,
            "frame_id": list(range(n)),
            "time_seconds": [i * 0.04 for i in range(n)],
            "is_ball": [True] * n,
            "x": [50.0] * n,
            "y": [34.0] * n,
            "z": list(z_values),
            "z_source": [z_source] * n,
        }
    )


def test_persistent_off_ground_measured_is_flagged():
    # SkillCorner signature: ~85% off-ground, declared measured -> inconsistent + warns.
    frames = _ball_frames([0.2] * 85 + [0.0] * 15, "measured")
    with pytest.warns(ZProvenanceWarning):
        d = validate_z_provenance(frames)
    assert d.consistent is False
    assert d.off_ground_fraction > 0.7


def test_measured_ground_signature_is_clean():
    # IDSSE: ~30% off-ground, p50 ~0, declared measured -> consistent (no warning).
    frames = _ball_frames([0.0] * 70 + [1.5] * 30, "measured")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZProvenanceWarning)
        d = validate_z_provenance(frames)
    assert d.consistent is True


def test_gradient_sports_negatives_are_not_flagged():
    # GS: negatives to -3.2 m, p50 ~0, ~45% off-ground, declared measured -> MUST be clean
    # (ground-level negatives are below the aerial boundary -- legitimately measured; §11.3).
    frames = _ball_frames([1.0] * 45 + [0.0] * 40 + [-3.2] * 15, "measured")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZProvenanceWarning)
        d = validate_z_provenance(frames)
    assert d.consistent is True


def test_all_flat_measured_is_flagged():
    # declared measured but z is dead (all ground) -> no live z -> flagged.
    frames = _ball_frames([0.0] * 100, "measured")
    with pytest.warns(ZProvenanceWarning):
        d = validate_z_provenance(frames)
    assert d.consistent is False


def test_estimated_declaration_is_never_a_measured_contradiction():
    # already-declared estimated -> the guard only polices "measured", so always consistent.
    frames = _ball_frames([0.2] * 85 + [0.0] * 15, "estimated")
    d = validate_z_provenance(frames, warn=False)
    assert d.consistent is True
    assert d.declared == "estimated"
