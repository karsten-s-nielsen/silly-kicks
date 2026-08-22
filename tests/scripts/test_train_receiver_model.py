"""Task 6: the SB360 public-receiver training driver."""

from __future__ import annotations

import json
import sys

import numpy as np
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
    """3 SB360-style completed passes: the intended receiver sits ON the release->reception ray (so the
    TRAJECTORY strategy labels it, statsbomb -> trajectory) with off-ray teammates as negatives."""
    P, S = _T["pass"], _R["success"]
    acts, frames = [], []
    aid, t, fid = 1, 10.0, game_id * 1000
    for k in range(3):
        receiver = 10 + k  # 10, 11, 12
        acts.append((aid, game_id, 1, t, 1, 9, P, S, 50, 34, 70, 34))  # pass toward (70,34)
        acts.append((aid + 1, game_id, 1, t + 0.5, 1, receiver, P, S, 70, 34, 30, 40))  # receiver's touch at (70,34)
        rows = [
            (True, pd.NA, pd.NA, False, 50.0, 34.0),
            (False, 9, 1, False, 50.0, 34.0),  # passer
            (False, receiver, 1, False, 70.0, 34.0),  # intended receiver -- ON the +x ray (perp 0)
            (False, 50, 1, False, 55.0, 55.0),  # off-ray teammate
            (False, 51, 1, False, 55.0, 12.0),  # off-ray teammate
            (False, 20, 2, False, 60.0, 34.0),
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


def _one_pass_match(*, receiver_in_frame: bool):
    """A single completed pass (real ids) whose next-touch receiver (player 10) is present or ABSENT in the
    frame -- for the id-strategy label / drop-on-no-match test (Q4)."""
    P, S = _T["pass"], _R["success"]
    rx = 10 if receiver_in_frame else 99
    acts = [
        (1, 7, 1, 10.0, 1, 9, P, S, 50, 34, 70, 34),
        (2, 7, 1, 10.5, 1, rx, P, S, 70, 34, 40, 34),  # receiver's next touch (id = rx)
    ]
    rows = [
        (True, pd.NA, pd.NA, False, 50.0, 34.0, 4.0, 0.0),
        (False, 9, 1, False, 50.0, 34.0, 0.0, 0.0),
        (False, 10, 1, False, 70.0, 34.0, 1.0, 0.0),
        (False, 11, 1, False, 55.0, 52.0, 0.0, 0.0),
        (False, 20, 2, False, 62.0, 34.0, 0.0, 0.0),
        (False, 30, 2, True, 100.0, 34.0, 0.0, 0.0),
    ]
    fr = pd.DataFrame(rows, columns=["is_ball", "player_id", "team_id", "is_goalkeeper", "x", "y", "vx", "vy"])
    fr["game_id"], fr["period_id"], fr["time_seconds"], fr["frame_id"] = 7, 1, 10.0, 700
    fr["source_provider"] = "gradientsports"
    return pd.DataFrame(acts, columns=_ACT_COLS), fr.astype({"player_id": "Int64", "team_id": "Int64"})


def _traj_match(*, receiver_on_ray: bool):
    """SB360-style SYNTHETIC-id frame (candidate ids never match the real action receiver id). Reception
    is on teammate 112's ray (on-ray) or up-field near nobody (ambiguous) -- for the trajectory-strategy
    label / drop-on-ambiguous test (Q1/Q2)."""
    P, S = _T["pass"], _R["success"]
    rec_x, rec_y = (70, 34) if receiver_on_ray else (50, 60)
    acts = [
        (1, 8, 1, 10.0, 1, 900, P, S, 50, 34, rec_x, rec_y),
        (2, 8, 1, 10.5, 1, 901, P, S, rec_x, rec_y, 30, 40),  # receiver real id 901, NOT in the frame
    ]
    rows = [
        (True, pd.NA, pd.NA, False, 50.0, 34.0),
        (False, 111, 1, False, 50.0, 34.0),  # passer (synthetic id)
        (False, 112, 1, False, 70.0, 34.0),  # on the +x ray
        (False, 113, 1, False, 55.0, 20.0),  # off-ray
        (False, 120, 2, False, 62.0, 34.0),
        (False, 130, 2, True, 100.0, 34.0),
    ]
    fr = pd.DataFrame(rows, columns=["is_ball", "player_id", "team_id", "is_goalkeeper", "x", "y"])
    fr["game_id"], fr["period_id"], fr["time_seconds"], fr["frame_id"] = 8, 1, 10.0, 800
    fr["source_provider"] = "statsbomb"
    return pd.DataFrame(acts, columns=_ACT_COLS), fr.astype({"player_id": "Int64", "team_id": "Int64"})


def test_labeling_strategy_for_provider():
    from scripts.train_receiver_model import labeling_strategy_for_provider as strat

    assert strat("statsbomb") == "trajectory"  # identity-less freeze frames
    assert strat("gradientsports") == "id"  # real tracking identity
    assert strat("skillcorner") == "id"


def test_id_strategy_labels_receiver_and_drops_off_frame_receiver():
    from scripts.train_receiver_model import extract_candidate_rows

    a, fr = _one_pass_match(receiver_in_frame=True)
    rows = extract_candidate_rows(a, fr, feature_set="public", labeling_strategy="id")
    assert int(rows["label"].sum()) == 1
    assert rows.loc[rows["label"] == 1, "candidate_id"].iloc[0] == "10"
    a2, fr2 = _one_pass_match(receiver_in_frame=False)  # receiver off-frame -> Q4 DROP, never trajectory-guess
    assert len(extract_candidate_rows(a2, fr2, feature_set="public", labeling_strategy="id")) == 0


def test_trajectory_strategy_labels_on_ray_and_drops_ambiguous():
    from scripts.train_receiver_model import extract_candidate_rows

    a, fr = _traj_match(receiver_on_ray=True)
    rows = extract_candidate_rows(a, fr, feature_set="public", labeling_strategy="trajectory")
    assert int(rows["label"].sum()) == 1
    assert rows.loc[rows["label"] == 1, "candidate_id"].iloc[0] == "112"  # the on-ray teammate
    a2, fr2 = _traj_match(receiver_on_ray=False)  # no teammate clearly on the ray -> Q2 DROP (not all-zero)
    assert len(extract_candidate_rows(a2, fr2, feature_set="public", labeling_strategy="trajectory")) == 0


def test_trajectory_winner_unit():
    from scripts.train_receiver_model import _LABEL_LANE_WIDTH_M, _trajectory_winner

    tm = pd.DataFrame({"player_id": [112, 113], "x": [70.0, 55.0], "y": [34.0, 20.0]}).astype({"player_id": "Int64"})
    rel = np.array([50.0, 34.0])
    assert _trajectory_winner(tm, rel, np.array([70.0, 34.0]), _LABEL_LANE_WIDTH_M) == "112"
    assert _trajectory_winner(tm, rel, np.array([50.0, 60.0]), _LABEL_LANE_WIDTH_M) is None  # ambiguous -> drop


def _cand_rows(games, *, invert=False):
    """Separable public candidate rows: the receiver (cid1) has HIGH space; ``invert`` gives the receiver
    LOW space instead -- a cleanly-OPPOSITE space->label mapping that must fail the pooling gate (the
    pooled model would then predict the wrong candidate on the primary held-out)."""
    rows = []
    for g in games:
        for act in range(6):
            for cid, is_rx in [(1, True), (2, False), (3, False)]:
                sp = (12.0 if is_rx else 4.0) if not invert else (4.0 if is_rx else 12.0)
                rows.append(
                    {
                        "ball_dist": 15.0,
                        "lane_pressure": 0.5,
                        "space": sp,
                        "candidate_id": str(cid),
                        "label": int(is_rx),
                        "game_id": g,
                        "action_id": act,
                        "n_candidates": 3,
                    }
                )
    return pd.DataFrame(rows)


def test_pooling_gate_keeps_agreeing_pool_and_drops_contradictory_pool():
    """Q3: the pool earns inclusion ONLY if it does not regress the PRIMARY held-out top-1."""
    from scripts.train_receiver_model import pooling_gate

    primary = _cand_rows([1, 2])
    keep = pooling_gate(primary, _cand_rows([3, 4]), "public")  # pool agrees -> no regression -> keep
    assert keep["keep_pool"] is True
    drop = pooling_gate(primary, _cand_rows([3, 4], invert=True), "public")  # OPPOSITE mapping -> regression
    assert drop["keep_pool"] is False
    assert set(drop) >= {"primary_only_top1", "pooled_top1", "margin", "keep_pool"}


def test_main_pooled_records_the_gate_and_coverage(tmp_path, monkeypatch):
    """Q3 end-to-end: --pool-provider extracts the pool (GS/id), runs the earned-inclusion gate, and
    RECORDS it + label coverage in the provenanced manifest. The primary (statsbomb/trajectory) is the
    serve target."""
    monkeypatch.setattr("scripts._loader_pining.load_statsbomb_matches", lambda *a, **k: iter([_match(1), _match(2)]))
    monkeypatch.setattr(
        "scripts._loader_pining.load_matches", lambda providers, cache_dir: iter([_gs_match(1), _gs_match(2)])
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
            "--provider",
            "statsbomb",
            "--pool-provider",
            "gradientsports",
            "--allow-dirty",
            "--min-rows",
            "1",
            "--min-passes",
            "1",
        ],
    )
    TRM.main()
    man = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert man["pooling_gate"]["pool_provider"] == "gradientsports"
    assert isinstance(man["pooling_gate"]["keep_pool"], bool)
    assert set(man["providers_trained"]) <= {"statsbomb", "gradientsports"}
    assert man["labeling_strategy"] == "trajectory"  # the primary's strategy
    assert "n_completed_passes" in man["label_coverage"] and "n_kept_passes" in man["label_coverage"]


def test_labels_read_the_outcome_but_features_do_not():
    """Driver-layer leakage guard (Q1 put the outcome-reader in the driver): perturbing the reception
    anchor (the next-touch start) MOVES the trajectory label but leaves the FEATURE columns byte-identical
    -- labels may read the outcome, features must not."""
    from scripts.train_receiver_model import extract_candidate_rows

    a, fr = _traj_match(receiver_on_ray=True)  # reception (70,34) -> winner 112
    base = extract_candidate_rows(a, fr, feature_set="public", labeling_strategy="trajectory")
    a2 = a.copy()
    a2.loc[a2["action_id"] == 2, ["start_x", "start_y"]] = [55.0, 20.0]  # move the reception toward teammate 113
    moved = extract_candidate_rows(a2, fr, feature_set="public", labeling_strategy="trajectory")
    feat_cols = ["candidate_id", "ball_dist", "lane_pressure", "space"]
    pd.testing.assert_frame_equal(base[feat_cols].reset_index(drop=True), moved[feat_cols].reset_index(drop=True))
    assert not base["label"].reset_index(drop=True).equals(moved["label"].reset_index(drop=True))  # label moved


def test_trajectory_excludes_actor_candidate_on_identity_less_frame():
    """A-F1: on an identity-less (SB360) frame the passer's real id can't match a synthetic frame id, so
    the ACTOR survives at the release, ~on the pass ray -- a spurious candidate that (if annotated forward)
    could win a trajectory label. It is excluded on identity-less frames; on an identity frame id-exclusion
    already removed the passer, so no position-exclusion fires. (Measured: on real SB360 the actor sits at
    the release exactly -- displacement 0.0 -- so this is data-cleanliness + defence, not an active bug.)"""
    from scripts.train_receiver_model import extract_candidate_rows

    a, fr = _traj_match(receiver_on_ray=True)  # synthetic passer id 111 at (50,34)=release; action passer id 900
    rows = extract_candidate_rows(a, fr, feature_set="public", labeling_strategy="trajectory")
    assert "111" not in set(rows["candidate_id"])  # the actor-at-release is excluded (identity-less)
    a2, fr2 = _one_pass_match(receiver_in_frame=True)  # real ids: passer id 9 excluded by id, not by position
    rows2 = extract_candidate_rows(a2, fr2, feature_set="public", labeling_strategy="id")
    assert "9" not in set(rows2["candidate_id"])


def test_empty_pool_does_not_flip_providers_trained(tmp_path, monkeypatch):
    """B-F1: a pool provider that yields nothing ties the gate (margin 0 -> keep_pool True) but contributed
    ZERO rows -- it must NOT be stamped into providers_trained / corpus_visibility."""
    monkeypatch.setattr("scripts._loader_pining.load_statsbomb_matches", lambda *a, **k: iter([_match(1), _match(2)]))
    monkeypatch.setattr("scripts._loader_pining.load_matches", lambda providers, cache_dir: iter([]))  # empty pool
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_receiver_model.py",
            "--out",
            str(tmp_path / "out"),
            "--shard-root",
            str(tmp_path / "sh"),
            "--provider",
            "statsbomb",
            "--pool-provider",
            "gradientsports",
            "--allow-dirty",
            "--min-rows",
            "1",
            "--min-passes",
            "1",
        ],
    )
    TRM.main()
    man = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert man["providers_trained"] == ["statsbomb"]  # GS contributed nothing -> not stamped


def test_empty_primary_exits_cleanly_not_keyerror(tmp_path, monkeypatch):
    """B-F2: a primary that produces no candidate rows (reconcile -> a column-less frame) must raise the
    'vacuous training set' SystemExit, not a KeyError on rows['label']."""
    monkeypatch.setattr(TRM, "_extract_provider_rows", lambda *a, **k: (pd.DataFrame(), {}))  # column-less empty
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
    with pytest.raises(SystemExit, match="vacuous training set"):
        TRM.main()


def _owner_cand_rows():
    """Pre-extracted owner (velocity) candidate rows: both classes, 2 namespaced games, separable."""
    rows = []
    for g in (1, 2):
        for a in range(4):
            for cid, is_rx in [(1, 1), (2, 0), (3, 0)]:
                rows.append(
                    {
                        "ball_dist": 15.0,
                        "lane_pressure": 0.5,
                        "space": 12.0 if is_rx else 4.0,
                        "release_dir_align": 0.9 if is_rx else 0.1,
                        "closing_speed": 3.0 if is_rx else 0.0,
                        "candidate_id": str(cid),
                        "label": is_rx,
                        "game_id": f"gradientsports:{g}",
                        "action_id": a,
                        "n_candidates": 3,
                    }
                )
    return pd.DataFrame(rows)


def test_owner_rows_skips_the_training_reparse(tmp_path, monkeypatch):
    """--owner-rows reads a pre-extracted parquet and SKIPS the training corpus re-parse (a GS match is ~4M
    frames). Only the deployment gate (mocked here) would touch the provider, so the training loader is
    never called."""
    rows_path = tmp_path / "candidate_rows.parquet"
    _owner_cand_rows().to_parquet(rows_path)
    called = {"load": False}

    def _boom(*a, **k):
        called["load"] = True
        return iter([])

    monkeypatch.setattr("scripts._loader_pining.load_matches", _boom)
    monkeypatch.setattr(TRM, "_resolve_deployment", lambda *a, **k: {"decisive": False, "margin": float("nan")})
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
            "--owner-rows",
            str(rows_path),
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
    assert "velocity_ablation" in man and "deployment" in man  # M-A resolution completed from the parquet
    assert called["load"] is False  # the training path did NOT re-parse the corpus


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        old = sys.argv
        sys.argv = ["train_receiver_model.py", "--help"]
        try:
            TRM.main()
        finally:
            sys.argv = old
    assert exc.value.code == 0
