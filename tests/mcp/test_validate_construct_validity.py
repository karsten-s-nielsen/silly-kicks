"""validate_construct_validity: memo reader surfaces recorded verdicts; xtgk fails loud; read-only."""

import pytest

from silly_kicks.mcp import _memo, server

PROVENANCED = ["gk_decision", "territorial_defense"]


@pytest.mark.parametrize("family", PROVENANCED)
def test_verdict_shape(family):
    r = server.validate_construct_validity(family)
    assert set(r) >= {"metric_family", "memo_path", "run_commit", "run_tree_dirty", "run_commit_is_head", "verdict"}
    assert r["run_tree_dirty"] is False


@pytest.mark.parametrize("family", PROVENANCED)
def test_thin_wrapper_equality(family):
    assert server.validate_construct_validity(family)["verdict"] == _memo.recorded_verdict(
        _memo.read_validity_memo(family)
    )


def test_xtgk_raises_until_provenance():
    with pytest.raises(ValueError):
        server.validate_construct_validity("xtgk_possession_value")


def test_fail_loud_on_absent_memo(tmp_path):
    with pytest.raises(FileNotFoundError):
        server.validate_construct_validity("gk_decision", research_root=tmp_path)


def test_read_only_on_dirty_tree(dirty_tree):
    server.validate_construct_validity("gk_decision")
    assert dirty_tree.no_new_artifacts()
