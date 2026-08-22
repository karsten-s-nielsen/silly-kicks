"""Task 6: the SB360 public-receiver training driver."""

from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

import scripts.train_receiver_model as TRM
from scripts.train_receiver_model import _R, _T

_ACT_COLS = [
    "action_id",
    "game_id",
    "period_id",
    "time_seconds",
    "team_id",
    "player_id",
    "type_id",
    "result_id",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
]


def _match(game_id: int):
    """3 completed passes (each followed by the receiver's own action) + a freeze frame per pass."""
    P, S = _T["pass"], _R["success"]
    acts, frames = [], []
    aid, t, fid = 1, 10.0, game_id * 1000
    for k in range(3):
        receiver = 10 + k  # 10, 11, 12
        acts.append((aid, game_id, 1, t, 1, 9, P, S, 50, 34, 60 + 3 * k, 34 + 6 * k))
        acts.append((aid + 1, game_id, 1, t + 0.5, 1, receiver, P, S, 60, 40, 30, 40))  # receiver's next touch
        rows = [
            (True, pd.NA, pd.NA, False, 50.0, 34.0),
            (False, 9, 1, False, 50.0, 34.0),
            (False, 10, 1, False, 70.0, 30.0),
            (False, 11, 1, False, 60.0, 52.0),
            (False, 12, 1, False, 62.0, 16.0),
            (False, 20, 2, False, 58.0, 34.0),
            (False, 21, 2, False, 55.0, 45.0),
            (False, 30, 2, True, 100.0, 34.0),
        ]
        fr = pd.DataFrame(rows, columns=["is_ball", "player_id", "team_id", "is_goalkeeper", "x", "y"])
        fr["game_id"], fr["period_id"], fr["time_seconds"], fr["frame_id"] = game_id, 1, t, fid
        fr["source_provider"] = "statsbomb"
        frames.append(fr.astype({"player_id": "Int64", "team_id": "Int64"}))
        aid, t, fid = aid + 2, t + 5.0, fid + 1
    actions = pd.DataFrame(acts, columns=_ACT_COLS)
    return ("statsbomb", game_id, actions, pd.concat(frames, ignore_index=True), 1, None)


def test_main_trains_bundle_with_provenance_and_m2_distribution(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts._loader_pining.load_statsbomb_matches", lambda *a, **k: iter([_match(1), _match(2)]))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_receiver_model.py",
            "--out",
            str(tmp_path / "out"),
            "--shard-root",
            str(tmp_path / "sh"),
            "--allow-dirty",
            "--min-rows",
            "1",
            "--min-passes",
            "1",
        ],
    )
    TRM.main()
    man = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert man["feature_set"] == "public" and man["n_matches"] == 2 and man["n_passes"] == 6
    assert man["run_commit"] and isinstance(man["run_tree_dirty"], bool)
    assert "mean" in man["candidate_count_distribution"]  # M2 shift-measurement
    # the bundle loads back through the full ADR-011 integrity path
    from silly_kicks.tracking._receiver import ReceiverModel

    m = ReceiverModel.load(tmp_path / "out" / "model")
    assert m.feature_set == "public"


def _gs_match(game_id: int):
    """A GS-style match: REAL tracking frames WITH velocity (the owner variant requires it), completed
    passes only. `--provider gradientsports` must route here via load_matches, NOT to SB360."""
    P, S = _T["pass"], _R["success"]
    acts, frames = [], []
    aid, t, fid = 1, 10.0, game_id * 1000
    for k in range(3):
        receiver = 10 + k
        acts.append((aid, game_id, 1, t, 1, 9, P, S, 50, 34, 60 + 3 * k, 34 + 6 * k))
        acts.append((aid + 1, game_id, 1, t + 0.5, 1, receiver, P, S, 60, 40, 30, 40))
        rows = [
            (True, pd.NA, pd.NA, False, 50.0, 34.0, 4.0, 0.0),  # ball moving +x
            (False, 9, 1, False, 50.0, 34.0, 0.0, 0.0),
            (False, 10, 1, False, 70.0, 30.0, 1.0, 0.0),
            (False, 11, 1, False, 60.0, 52.0, 0.0, 1.0),
            (False, 12, 1, False, 62.0, 16.0, 0.0, -1.0),
            (False, 20, 2, False, 58.0, 34.0, 0.0, 0.0),
            (False, 21, 2, False, 55.0, 45.0, 0.0, 0.0),
            (False, 30, 2, True, 100.0, 34.0, 0.0, 0.0),
        ]
        fr = pd.DataFrame(rows, columns=["is_ball", "player_id", "team_id", "is_goalkeeper", "x", "y", "vx", "vy"])
        fr["game_id"], fr["period_id"], fr["time_seconds"], fr["frame_id"] = game_id, 1, t, fid
        fr["source_provider"] = "gradientsports"
        frames.append(fr.astype({"player_id": "Int64", "team_id": "Int64"}))
        aid, t, fid = aid + 2, t + 5.0, fid + 1
    return ("gradientsports", game_id, pd.DataFrame(acts, columns=_ACT_COLS), pd.concat(frames, ignore_index=True), 1)


def test_load_corpus_routes_owner_provider_to_tracking_loader(monkeypatch):
    """The owner (GS) variant needs REAL tracking frames with velocity, so --provider gradientsports must
    load via load_matches, NOT load_statsbomb_matches (velocity-less SB360 would crash owner extraction)."""
    seen = {}

    def fake_matches(providers, cache_dir):
        seen["matches"] = providers
        return iter([])

    def fake_sb(*a, **k):
        seen["sb"] = True
        return iter([])

    monkeypatch.setattr("scripts._loader_pining.load_matches", fake_matches)
    monkeypatch.setattr("scripts._loader_pining.load_statsbomb_matches", fake_sb)
    list(TRM._load_corpus("gradientsports", None))
    assert seen.get("matches") == ["gradientsports"] and "sb" not in seen
    list(TRM._load_corpus("statsbomb", None))
    assert seen.get("sb") is True


def test_owner_run_records_m_a_resolution(tmp_path, monkeypatch):
    """The owner (GS) variant records the M-A resolution in its provenanced manifest: velocity ablation
    (i, real) + -- given a public bundle -- the deployment gate (ii). Also pins that owner extraction runs
    on velocity frames end-to-end."""
    monkeypatch.setattr(
        "scripts._loader_pining.load_matches", lambda providers, cache_dir: iter([_gs_match(1), _gs_match(2)])
    )
    monkeypatch.setattr(
        TRM, "_resolve_deployment", lambda *a, **k: {"decisive": False, "margin": float("nan"), "n_scored": 0}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_receiver_model.py",
            "--out",
            str(tmp_path / "out"),
            "--shard-root",
            str(tmp_path / "sh"),
            "--feature-set",
            "owner",
            "--provider",
            "gradientsports",
            "--public-bundle",
            str(tmp_path / "pub"),
            "--allow-dirty",
            "--min-rows",
            "1",
            "--min-passes",
            "1",
        ],
    )
    TRM.main()
    man = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert man["feature_set"] == "owner" and man["provider"] == "gradientsports"
    assert "velocity_ablation" in man and "velocity_delta" in man["velocity_ablation"]  # M-A(i) real
    assert man["deployment"]["decisive"] is False  # M-A(ii) wiring reached with --public-bundle


def test_resolve_deployment_runs_the_real_second_pass(tmp_path, monkeypatch):
    """Exercise the REAL M-A(ii) plumbing (for_each -> _deployment_counts_for_match -> reconcile -> pool),
    not the mocked wiring. A completed-only corpus scores nothing on the FAILED subset -> non-decisive,
    but the second sharded pass runs end-to-end and writes its counts parquet."""
    import numpy as np

    from silly_kicks.tracking._receiver import ReceiverModel

    X = pd.DataFrame({"ball_dist": [15.0] * 8, "lane_pressure": [0.0, 1.5] * 4, "space": [12.0, 4.0] * 4})
    y = np.array([1, 0] * 4)
    pub = ReceiverModel("public").fit(X, y)
    pub.save(tmp_path / "pub")
    own = ReceiverModel("public").fit(X, y)  # a positions-only stand-in is enough for the plumbing
    monkeypatch.setattr("scripts._loader_pining.load_matches", lambda providers, cache_dir: iter([_gs_match(1)]))
    out = tmp_path / "out"
    out.mkdir()
    decision = TRM._resolve_deployment(tmp_path / "pub", own, "gradientsports", None, tmp_path / "sh", out, "abc123")
    assert {"decisive", "margin", "public_top1", "owner_top1", "coverage"} <= set(decision)
    assert decision["decisive"] is False  # completed-only corpus -> nothing scored on the failed subset
    assert (out / "deployment_counts.parquet").exists()  # the second pass persisted its shards


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        old = sys.argv
        sys.argv = ["train_receiver_model.py", "--help"]
        try:
            TRM.main()
        finally:
            sys.argv = old
    assert exc.value.code == 0
