"""The memo-read seam surfaces a recorded verdict, and fails loud on any untrustworthy memo."""

import json

import pytest

from silly_kicks.mcp import _memo


def test_reads_recorded_memo():
    v = _memo.read_validity_memo("gk_decision")
    assert "verdicts" in v and v["run_tree_dirty"] is False


def test_raises_on_absent_memo(tmp_path):
    with pytest.raises(FileNotFoundError):
        _memo.read_validity_memo("gk_decision", research_root=tmp_path)


def test_raises_on_dirty_provenance(tmp_path):
    d = tmp_path / "gk_decision_construct_validity"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({"run_commit": "x", "run_tree_dirty": True, "verdicts": {}}))
    with pytest.raises(ValueError):
        _memo.read_validity_memo("gk_decision", research_root=tmp_path)


def test_raises_on_missing_provenance(tmp_path):
    d = tmp_path / "xtgk_possession_value"
    d.mkdir(parents=True)
    (d / "gate.json").write_text(json.dumps({"wc2022": {"authorising": True}}))  # no run_commit/run_tree_dirty
    with pytest.raises(ValueError):
        _memo.read_validity_memo("xtgk_possession_value", research_root=tmp_path)


def test_committed_xtgk_memo_lacks_provenance():
    # The real committed gate.json has no provenance -> the tool fails loud until it is regenerated.
    with pytest.raises(ValueError):
        _memo.read_validity_memo("xtgk_possession_value")
