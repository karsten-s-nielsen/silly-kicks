"""The receiver widening gate (combined-cycle-completion spec section 7): scoring rule, exclusions,
identification with its negative control, decision rule, no ids in the output."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts import validate_receiver_widening as G
from silly_kicks.tracking._receiver import ReceiverModel


def _rows(n_games=12, passes=30, seed=0):
    """Synthetic candidate rows in the trainer's public schema: 4 candidates per pass, one labelled."""
    from scripts.train_receiver_model import _feature_names

    rng = np.random.default_rng(seed)
    names = _feature_names("public")
    recs = []
    for g in range(n_games):
        for a in range(passes):
            for c in range(4):
                feats = rng.standard_normal(len(names))
                recs.append(
                    {
                        "game_id": f"statsbomb:{g}",
                        "action_id": a,
                        "label": int(c == 0),
                        **dict(zip(names, feats + (1.5 if c == 0 else 0.0), strict=True)),
                    }
                )
    return pd.DataFrame(recs)


def test_per_pass_hits_matches_the_trainers_top1():
    from scripts.train_receiver_model import _feature_names, _top1_accuracy

    rows = _rows()
    m = ReceiverModel("public").fit(rows[_feature_names("public")], rows["label"])
    hits = G.per_pass_hits(m, rows)
    assert abs(hits["hit"].mean() - _top1_accuracy(m, rows, "public")) < 1e-12
    assert len(hits) == rows.groupby(["game_id", "action_id"]).ngroups


def test_identical_models_give_zero_difference_and_the_q3_rule_passes():
    from scripts.train_receiver_model import _feature_names

    rows = _rows()
    committed = ReceiverModel("public").fit(rows[_feature_names("public")], rows["label"])
    out = G.gate(rows, committed, exclude=[], n_boot=200)
    assert out["n_test_matches"] == 12
    assert out["decision_rule"] == G.DECISION_RULE
    # new is a per-fold refit, old the full fit: both near-perfect on this separable fixture
    assert abs(out["diff_new_minus_old"]) < 0.05


def test_exclusions_are_removed_from_every_test_fold():
    from scripts.train_receiver_model import _feature_names

    rows = _rows()
    committed = ReceiverModel("public").fit(rows[_feature_names("public")], rows["label"])
    out = G.gate(rows, committed, exclude=["statsbomb:0", "statsbomb:1"], n_boot=50)
    assert out["n_test_matches"] == 10 and out["excluded_from_test"] == 2


def test_identification_succeeds_on_the_true_training_set_and_its_negative_control_fails():
    from scripts.train_receiver_model import _feature_names

    rows = _rows(n_games=24, seed=1)
    true_ids = [f"statsbomb:{g}" for g in range(6)]
    sub = rows[rows["game_id"].isin(true_ids)]
    committed = ReceiverModel("public").fit(sub[_feature_names("public")], sub["label"])
    G_ID_TOP1 = G.cv_top1(sub, "public")[0]
    report, exclude = G.identify(
        rows, [i.split(":")[1] for i in true_ids], committed, expected_top1=G_ID_TOP1, n_controls=5
    )
    assert report["identified"] is True and sorted(exclude) == sorted(true_ids)
    assert report["n_controls_passing"] == 0  # random 6-match subsets must not reproduce the committed fit


def test_output_never_records_a_match_id(tmp_path, monkeypatch):
    from scripts.train_receiver_model import _feature_names

    rows = _rows()
    raw = {f"statsbomb:{g}": f"38570{g:02d}" for g in range(12)}  # distinctive 7-digit ids, as SB360 uses
    rows.assign(game_id=rows["game_id"].map(raw)).to_parquet(
        tmp_path / "rows.parquet"
    )  # raw ids, as the trainer writes
    committed = ReceiverModel("public").fit(rows[_feature_names("public")], rows["label"])
    monkeypatch.setattr(G, "_committed_model", lambda: committed)
    monkeypatch.setattr(G, "_statsbomb_first30", lambda: [raw[f"statsbomb:{g}"] for g in range(3)])
    prov = {"commit": "0" * 40, "dirty": False, "tree_state": "clean"}
    G.run(tmp_path / "rows.parquet", tmp_path / "out", prov=prov, n_boot=50, n_controls=2)
    text = (tmp_path / "out" / "receiver_gate.json").read_text(encoding="utf-8")
    doc = json.loads(text)
    assert doc["run_commit"] == "0" * 40 and doc["run_tree_dirty"] is False
    assert "statsbomb:" not in text
    assert not any(mid in text for mid in raw.values())


@pytest.mark.parametrize(
    ("diff", "lb", "want"),
    [
        (0.0, -0.005, True),  # point exactly 0 passes (>=), LB inside the margin
        (0.004, -0.0099, True),  # LB just inside -0.01
        (0.004, float(np.nextafter(-0.01, 0.0)), True),  # the closest LB inside the bound
        (0.004, -0.01, False),  # LB AT -0.01: the bound is strict (B r4 CCC-PLAN-30)
        (0.004, -0.0101, False),  # LB just outside -0.01 (noisy pass blocked)
        (-0.001, -0.005, False),  # point estimate negative (Q3 rule)
        (float(np.nextafter(0.0, -1.0)), 0.002, False),  # the closest negative point fails
        (0.01, 0.002, True),
        (0.01, float("nan"), False),  # an undefined bound never ships
    ],
)
def test_decision_rule_combined(diff, lb, want):
    assert G.decide(diff, lb) is want  # D8 (owner-approved): point >= 0 AND 95% LB > -0.01


def test_the_d8_margin_and_rule_label_are_pinned():
    assert G.MARGIN == 0.01 and G.DECISION_RULE == "point_estimate_ge_0_and_boot_lb95_gt_-0.01"


def test_identification_fails_when_the_negative_control_stops_discriminating(monkeypatch):
    """B r4 CCC-PLAN-30: if random subsets reproduce the committed fit too, the candidate proves nothing."""
    calls = []

    def always(rows, ids, committed, expected_top1):
        calls.append(tuple(ids))
        return True, {"n_present": len(ids)}

    monkeypatch.setattr(G, "_reproduces", always)
    rows = pd.DataFrame({"game_id": [f"statsbomb:{g}" for g in range(12)]})
    report, exclude = G.identify(rows, ["0", "1", "2"], committed=None, n_controls=4)
    assert len(calls) == 1 + 4  # the candidate plus every control was actually tried
    assert report["candidate_reproduces"] is True and report["n_controls_passing"] == 4
    assert report["identified"] is False and exclude == []


def test_identification_fails_when_the_candidate_does_not_reproduce(monkeypatch):
    monkeypatch.setattr(G, "_reproduces", lambda rows, ids, committed, expected_top1: (False, {"n_present": len(ids)}))
    rows = pd.DataFrame({"game_id": [f"statsbomb:{g}" for g in range(12)]})
    report, exclude = G.identify(rows, ["0", "1", "2"], committed=None, n_controls=3)
    assert report["candidate_reproduces"] is False and report["identified"] is False and exclude == []


def test_the_wrong_match_set_does_not_reproduce_the_committed_fit():
    """Real reproduction (no stub): the committed model was fit on games 0-5; games 6-11 must not identify."""
    from scripts.train_receiver_model import _feature_names

    rows = _rows(n_games=24, seed=1)
    true_ids = [f"statsbomb:{g}" for g in range(6)]
    sub = rows[rows["game_id"].isin(true_ids)]
    committed = ReceiverModel("public").fit(sub[_feature_names("public")], sub["label"])
    top1 = G.cv_top1(sub, "public")[0]
    report, exclude = G.identify(rows, [str(g) for g in range(6, 12)], committed, expected_top1=top1, n_controls=2)
    assert report["candidate_reproduces"] is False and report["identified"] is False and exclude == []
