"""The idsse fixtures reach the converter through ``shape_events_to_native``, read in one place.

No test file other than ``tests/_idsse_fixtures.py`` may build a path to ``per_period_match.parquet`` or
``sample_match.parquet`` under ``tests/datasets/idsse``, so the short bronze-to-converter path cannot come back.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parent
_HELPER = _TESTS / "_idsse_fixtures.py"
# a path expression that joins the idsse fixture directory and one of the two file names
_IDSSE_PATH = re.compile(
    r"""(["']idsse["']\s*/\s*["'](per_period_match|sample_match)\.parquet["'])"""
    r"""|(_IDSSE\w*\s*/\s*["'](per_period_match|sample_match)\.parquet["'])"""
    r"""|(datasets/idsse/(per_period_match|sample_match)\.parquet["'])"""
)


def test_only_the_helper_reads_the_idsse_fixtures():
    offenders = sorted(
        str(p.relative_to(_TESTS))
        for p in _TESTS.rglob("*.py")
        if p not in (_HELPER, Path(__file__).resolve()) and _IDSSE_PATH.search(p.read_text(encoding="utf-8"))
    )
    assert offenders == [], f"read the idsse fixtures through tests/_idsse_fixtures.py: {offenders}"


def test_the_pin_has_teeth():
    assert _IDSSE_PATH.search('_REPO_ROOT / "tests" / "datasets" / "idsse" / "sample_match.parquet"')
    assert _IDSSE_PATH.search('pd.read_parquet(_IDSSE_DIR / "per_period_match.parquet")')
    assert not _IDSSE_PATH.search('_REPO_ROOT / "tests" / "datasets" / "metrica" / "sample_match.parquet"')
    assert _IDSSE_PATH.search(_HELPER.read_text(encoding="utf-8")) is None  # the helper builds it from parts


def test_the_helper_gives_the_native_frame():
    from tests._idsse_fixtures import idsse_fixture_native

    native = idsse_fixture_native("sample_match")
    assert len(native) == 308 and "team" in native.columns


@pytest.mark.parametrize("loader", ["load_sportec_native", "load_sportec_native_per_period"])
def test_through_native_shaping_every_action_has_a_team_and_a_player(loader):
    # What one helper makes true (on the short bronze path, 17 / 116 actions had team "unknown" and 17 / 115 no player)
    from tests.invariants import _loaders

    actions, _home = getattr(_loaders, loader)()
    assert len(actions) > 0  # non-vacuity
    assert int((actions["team_id"].astype(str) == "unknown").sum()) == 0
    assert int(actions["player_id"].isna().sum()) == 0
