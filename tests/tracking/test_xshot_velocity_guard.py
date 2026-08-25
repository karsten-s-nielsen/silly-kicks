"""Task 1 (ADR-054): velocity-availability guard for ``compute_xshot_occurrence``.

``XSHOT_FEATURE_NAMES_FAITHFUL`` includes ``speed``; on a velocity-less SB360 freeze-frame that
feature is NaN and XGBoost's missing-value routing would fabricate a probability (the ADR-053
fabrication shape). The guard mirrors ``compute_xcross_attempt``'s two-prong contract at the shared
seam: DECLARED-unavailable (``speed_source == "unavailable"`` on every row) -> honest NaN; undeclared
missing ``vx``/``vy`` -> loud ``ValueError`` naming ``derive_velocities()``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import _xshot_occurrence as xs
from silly_kicks.tracking.schema import SPEED_SOURCE_UNAVAILABLE

# Reuse the real happy-path scoring fixture (both teams, GKs, a clear team-2 possession near x=20).
# Reusing it -- rather than a hand-built minimal frame that can NaN for possession/goal-resolution
# reasons -- is what keeps the non-vacuity assertion below honest ("a gate is only as good as the
# rows it scores").
from tests.tracking.test_xshot_occurrence import _synthetic_match_frames


def _declared_unavailable(frames: pd.DataFrame) -> pd.DataFrame:
    """The SB360 freeze-frame shape: drop vx/vy, mark EVERY row speed_source=unavailable."""
    out = frames.drop(columns=[c for c in ("vx", "vy") if c in frames.columns]).copy()
    out["speed"] = np.nan
    out["speed_source"] = SPEED_SOURCE_UNAVAILABLE
    return out


def _undeclared_missing_velocity(frames: pd.DataFrame) -> pd.DataFrame:
    """The forgot-``derive_velocities()`` shape: no vx/vy, but NOT declared unavailable
    (``_synthetic_match_frames`` leaves speed_source='native')."""
    return frames.drop(columns=[c for c in ("vx", "vy") if c in frames.columns]).copy()


def test_declared_velocity_unavailable_returns_nan_not_fabricated(monkeypatch):
    # With the position_only variant BUNDLED (commit 2), a declared-velocity-less frame now serves a
    # value via that variant (the SB360 unlock; asserted on real weights in test_position_only_bundled).
    # This guard remains the FALLBACK contract: with NO position_only variant available, the FAITHFUL
    # model must NOT be used (its `speed` feature -> XGBoost missing-value routing = the ADR-053
    # fabrication shape); auto-select falls back to honest NaN. Force the unbundled path to keep it covered.
    def _boom(cls, v):
        if v == "position_only":
            raise FileNotFoundError
        raise AssertionError("must NOT fall back to the faithful default on velocity-less frames")

    monkeypatch.setattr(xs.XShotOccurrenceModel, "from_variant", classmethod(_boom))
    frames = _declared_unavailable(_synthetic_match_frames(n_frames=5))
    with pytest.warns(UserWarning, match="position_only"):
        out = xs.compute_xshot_occurrence(frames, model=None, home_team_id=1)
    assert out["xshot_occurrence"].isna().all()


def test_undeclared_missing_velocity_raises_naming_remedy():
    frames = _undeclared_missing_velocity(_synthetic_match_frames(n_frames=5))
    with pytest.raises(ValueError, match="derive_velocities"):
        xs.compute_xshot_occurrence(frames, model=None, home_team_id=1)


def test_velocity_bearing_frame_is_scored_not_nan():
    # NON-VACUITY for the declared-NaN test: the SAME scoring context WITH velocity scores finite,
    # so the NaN above is SUPPRESSION, not "nothing scores here".
    frames = _synthetic_match_frames(n_frames=5)
    out = xs.compute_xshot_occurrence(frames, model=None, home_team_id=1)
    assert out["xshot_occurrence"].notna().any()
