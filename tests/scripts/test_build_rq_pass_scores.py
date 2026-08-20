"""Task 3/4: the corpus-pass driver `scripts/build_rq_pass_scores.py`."""

from __future__ import annotations

from scripts import build_rq_pass_scores as B


def test_shard_has_declared_columns(mini_actions, mini_frames):
    out = B.score_match(mini_actions, mini_frames)
    # keys the rows ACTUALLY carry, not a selection to the declaration
    assert set(out.columns) == set(B._EMITTED_SHARD_COLUMNS)
    assert (out["p_blocked_mean"] >= 0.0).all()  # p_blocked is an unbounded blocking intensity, not [0,1]
    assert (out["p_received_center"] >= 0.0).all()  # per-lane survival, stored for the margin score
    assert out["n_blocked"].isin([0, 1, 2, 3]).all()  # the per-lane p_blocked>p_received count
    assert out["control"].between(0.0, 1.0).all()


def test_lane_scoring_non_degenerate(mini_actions, mini_frames):
    out = B.score_match(mini_actions, mini_frames)
    # action 0: away defender 20 at (45,34) sits on the (30,34)->(58,34) lane -> non-zero block
    a0 = out[out["action_id"] == 0].iloc[0]
    assert a0["p_blocked_max"] > 0.0


def test_main_writes_pass_scores_and_stamps_provenance(tmp_path, monkeypatch, mini_actions, mini_frames):
    import json
    import sys

    def fake_load(**kw):  # one (provider, match_id, actions, frames, home) tuple
        yield ("gradientsports", "m1", mini_actions, mini_frames, 1)

    monkeypatch.setattr(B, "load_matches", fake_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_rq_pass_scores.py",
            "--out",
            str(tmp_path / "out"),
            "--shard-root",
            str(tmp_path / "sh"),
            "--allow-dirty",
            "--min-passes",
            "1",
            "--min-completed",
            "1",
        ],
    )
    B.main()
    man = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert man["schema"] == "rq-scores-2" and man["n_passes"] > 0
    assert isinstance(man["run_tree_dirty"], bool) and man["run_commit"]


def test_help_parses_and_exits_zero():
    import sys

    import pytest

    with pytest.raises(SystemExit) as exc:
        old = sys.argv
        sys.argv = ["build_rq_pass_scores.py", "--help"]
        try:
            B.main()
        finally:
            sys.argv = old
    assert exc.value.code == 0
