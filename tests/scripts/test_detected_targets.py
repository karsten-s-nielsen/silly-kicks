"""Detected-keeper training targets (spec 4.3)."""

import pandas as pd
import pytest

from silly_kicks.tracking._ghost_gk import keeper_detection_mask


def test_undetected_keeper_rows_are_dropped_for_skillcorner():
    """KILL-LINE: delete the `keep = keeper_detection_mask(...)` filter in the extractor loop
    -> this MUST fail (the undetected row survives)."""
    vis = pd.Series([True, False, True])
    keep = keeper_detection_mask(vis, provider="skillcorner")
    assert keep.sum() == 2


def test_a_provider_whose_flag_was_discarded_RAISES_rather_than_training_on_the_interpolator():
    with pytest.raises(ValueError, match=r"discarded|null"):
        keeper_detection_mask(pd.Series([None, None]), provider="skillcorner")
