"""Ghost-GK must expose WHICH keeper each label belongs to, and whether he was SEEN (spec 4.3).

Without this, keeper-grouped CV cannot be built and 'detected-keeper targets only' cannot be
enforced -- and ~80% of SkillCorner keeper positions are interpolator output.
"""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._ghost_gk import keeper_detection_mask


def test_detection_aware_provider_with_null_visibility_RAISES():
    """FAIL-CLOSED (spec 4.3). A null here means the kloppy gateway discarded the flag -- NOT
    that the keeper was observed. Reading it as 'keep' is the licensing landmine's failure shape."""
    with pytest.raises(ValueError, match="skillcorner"):
        keeper_detection_mask(pd.Series([None, None]), provider="skillcorner")


def test_detection_aware_provider_keeps_only_detected_keepers():
    mask = keeper_detection_mask(pd.Series([True, False, True]), provider="skillcorner")
    assert list(mask) == [True, False, True]


def test_fully_observed_provider_keeps_everything():
    """GS/IDSSE are full-pitch products: every player is observed, and no flag exists."""
    mask = keeper_detection_mask(pd.Series([None, None]), provider="gradientsports")
    assert list(mask) == [True, True]


def test_unknown_provider_RAISES():
    """Unknown providers are not assumed observed."""
    with pytest.raises(ValueError, match="unknown"):
        keeper_detection_mask(pd.Series([None]), provider="mystery_vendor")


def test_meta_gk_player_id_aligns_with_its_label():
    """The MANDATORY alignment check (plan Known-risks #2).

    The unit tests above cannot catch a length-preserving MISalignment of ``meta`` vs
    ``labels`` -- if a filter is applied to features/labels but not (identically) to meta,
    keeper identity silently drifts onto the wrong row. Build a fixture with TWO distinct
    keepers across several frames whose goal-relative *y* differs by identity (home GK at
    gr-y 30, away GK at gr-y 40), then assert every ``meta.gk_player_id`` carries the label
    of the keeper it names.
    """
    from silly_kicks.tracking import prepare_ghost_gk_training_data
    from tests.tracking.test_ghost_gk import _make_ghost_gk_frames

    parts = []
    for fid in range(1, 6):
        fr = _make_ghost_gk_frames(frame_id=fid, timestamp=float(fid))
        # Give the two keepers DISTINCT goal-relative y so identity is checkable.
        # Only x flips under the goal-relative transform; y is preserved, so the raw y IS
        # the goal-relative y for both ends. Home GK "p1" -> gr-y 30; away GK "a1" -> gr-y 40.
        fr.loc[(fr["player_id"] == "p1") & fr["is_goalkeeper"], "y"] = 30.0
        fr.loc[(fr["player_id"] == "a1") & fr["is_goalkeeper"], "y"] = 40.0
        parts.append(fr)
    frames = pd.concat(parts, ignore_index=True)

    features, labels, meta = prepare_ghost_gk_training_data(frames, home_team_id=1, return_meta=True)

    # All three shapes agree, and both filters left meta on a clean RangeIndex (a forgotten
    # reset_index would surface here as a misaligned .loc join below).
    assert len(features) == len(labels) == len(meta)
    assert list(meta.index) == list(range(len(meta)))
    assert list(features.index) == list(range(len(features)))
    assert "gk_player_id" in meta.columns
    assert "gk_visibility" in meta.columns

    # Identity <-> label: the row whose keeper is "p1" carries the y=30 label; "a1" -> y=40.
    for pid, expected_y in (("p1", 30.0), ("a1", 40.0)):
        sub = meta[meta["gk_player_id"] == pid]
        assert len(sub) > 0, f"no rows for keeper {pid}"
        assert np.allclose(labels.loc[sub.index, "gk_y"].to_numpy(), expected_y)


def test_prepare_default_return_shape_is_still_two_tuple():
    """Backcompat: default return_meta=False keeps the documented 2-tuple (four call sites)."""
    from silly_kicks.tracking import prepare_ghost_gk_training_data
    from tests.tracking.test_ghost_gk import _make_ghost_gk_frames

    frames = pd.concat(
        [_make_ghost_gk_frames(frame_id=fid, timestamp=float(fid)) for fid in range(1, 4)],
        ignore_index=True,
    )
    result = prepare_ghost_gk_training_data(frames, home_team_id=1)
    assert isinstance(result, tuple) and len(result) == 2
