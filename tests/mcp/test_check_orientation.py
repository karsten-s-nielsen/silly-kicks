"""check_orientation: verdict shape, both-sides geometry non-vacuity, untracked-GK, fail-loud, read-only."""

import pytest

from silly_kicks.mcp import server


def test_verdict_shape(fixture_ref):
    r = server.check_orientation(fixture_ref)
    assert set(r) >= {"match_ref", "provider", "direction_resolved", "geometry_agrees", "verdict", "measure"}
    assert r["verdict"] in {"OK", "UNORIENTED", "MISMATCH"}


def test_ok_on_correct(fixture_ref):  # non-vacuity: correct side (REAL slice)
    r = server.check_orientation(fixture_ref)
    assert r["verdict"] == "OK" and r["geometry_agrees"] is True


def test_mismatch_on_mislabeled(mislabeled_ref):  # non-vacuity: wrong side
    r = server.check_orientation(mislabeled_ref)
    assert r["verdict"] == "MISMATCH" and r["geometry_agrees"] is False


def test_unoriented(fixture_ref, monkeypatch):
    # Real RC4 (unlabelled_fraction=1.0) is a network-only skillcorner match; the recorded memo shape
    # (docs/research/adr028_rc4_orientation) is used to exercise the UNORIENTED branch offline.
    import measure_rc4_orientation as m

    rc4 = {
        "match_id": "1886347",
        "n_frames": 1,
        "player_rows": 22,
        "unlabelled_fraction": 1.0,
        "distinct_labels": [],
        "n_actions": 0,
        "n_flip_true": 0,
        "flip_true_fraction": 0.0,
        "orientation_warnings": 0,
    }
    monkeypatch.setattr(m, "measure", lambda loaded: rc4)
    assert server.check_orientation(fixture_ref)["verdict"] == "UNORIENTED"


def test_no_false_mismatch_untracked_home_gk(untracked_home_gk_ref):  # r6 SPEC-10 / r7 CONSIDER
    assert server.check_orientation(untracked_home_gk_ref)["verdict"] != "MISMATCH"


def test_thin_wrapper_passthrough(fixture_ref, _server_load):
    import measure_rc4_orientation as m

    loaded = _server_load["OK"]
    assert server.check_orientation(fixture_ref)["measure"] == server._json_safe(m.measure(loaded))


def test_fail_loud_on_bad_load(_server_load):
    with pytest.raises(RuntimeError):
        server.check_orientation("NOT_IN_REGISTRY")


def test_read_only_on_dirty_tree(dirty_tree, fixture_ref):
    server.check_orientation(fixture_ref)
    assert dirty_tree.no_new_artifacts()
