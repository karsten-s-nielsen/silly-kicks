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


def test_assert_emitted_schema_rejects_drift():
    """M5: the shard-schema check compares the keys the rows ACTUALLY carry, so a dropped/renamed/added
    key FAILS at the first shard -- it is NOT a selection to the declaration (which would mask drift)."""
    import pandas as pd
    import pytest

    ok = pd.DataFrame([{c: 0 for c in B._EMITTED_SHARD_COLUMNS}])
    B._assert_emitted_schema(ok, B._EMITTED_SHARD_COLUMNS)  # exact match: no raise
    dropped = pd.DataFrame([{c: 0 for c in B._EMITTED_SHARD_COLUMNS if c != "control"}])
    with pytest.raises(AssertionError):
        B._assert_emitted_schema(dropped, B._EMITTED_SHARD_COLUMNS)
    extra = pd.DataFrame([{**{c: 0 for c in B._EMITTED_SHARD_COLUMNS}, "surprise": 1}])
    with pytest.raises(AssertionError):
        B._assert_emitted_schema(extra, B._EMITTED_SHARD_COLUMNS)


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


def test_main_deleak_wiring_runs_with_receiver_model(tmp_path, monkeypatch, mini_actions, mini_frames):
    """L1: exercise the --receiver-model wiring end-to-end (from_variant -> threaded into score_match ->
    the receiver_model generation token). The de-leak SEMANTICS are unit-tested in test_rq_corpus_deleak;
    this pins that the driver path itself runs and writes output."""
    import json
    import sys

    import numpy as np
    import pandas as pd

    from silly_kicks.tracking._receiver import ReceiverModel

    intended = pd.DataFrame({"ball_dist": [15.0] * 12, "lane_pressure": [0.0] * 12, "space": [12.0] * 12})
    others = pd.DataFrame({"ball_dist": [15.0] * 12, "lane_pressure": [1.5] * 12, "space": [4.0] * 12})
    fitted = ReceiverModel("public").fit(
        pd.concat([intended, others], ignore_index=True), np.array([1] * 12 + [0] * 12)
    )
    monkeypatch.setattr(ReceiverModel, "from_variant", classmethod(lambda cls, key: fitted))

    def fake_load(**kw):
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
            "--receiver-model",
            "gs_owner",
            "--allow-dirty",
            "--min-passes",
            "1",
            "--min-completed",
            "1",
        ],
    )
    B.main()
    man = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert man["n_passes"] > 0  # the --receiver-model wiring executed without error and scored passes


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
