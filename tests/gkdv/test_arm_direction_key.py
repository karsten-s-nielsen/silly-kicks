"""TF-19 A+2 Task 4: the arm OUTPUT-column -> EXPECTED_DIRECTION key bridge.

The threat arm's OUTPUT column is ``delta_threat_suppression`` (``_arms.py``) but its registered
direction key is ``delta_threat`` (``EXPECTED_DIRECTION``). Without the bridge a §6.2 sign check on
the arm column would ``KeyError`` on the threat arm and silently skip its sign check -- so the bridge
is what lets EVERY arm column resolve, and an UNMAPPED arm must RAISE (never silently pass).
"""

from __future__ import annotations

import pytest

import silly_kicks.gkdv as gkdv
from silly_kicks.gkdv._validate import expected_direction_for_arm


def test_every_arm_column_resolves_to_expected_direction():
    # both physics arms are ATTACKER-value, so a deterrent keeper reads NEGATIVE on both.
    for arm_col in ("delta_das", "delta_threat_suppression"):
        assert expected_direction_for_arm(arm_col) == "negative"


def test_threat_arm_output_column_differs_from_its_direction_key():
    # the load-bearing bridge: the arm column is not the EXPECTED_DIRECTION key.
    assert "delta_threat_suppression" not in gkdv.EXPECTED_DIRECTION
    assert expected_direction_for_arm("delta_threat_suppression") == gkdv.EXPECTED_DIRECTION["delta_threat"]


def test_unmapped_arm_raises_not_silent():
    with pytest.raises(KeyError):
        expected_direction_for_arm("delta_not_an_arm")


def test_expected_direction_for_arm_is_exported_public():
    assert gkdv.expected_direction_for_arm is expected_direction_for_arm
