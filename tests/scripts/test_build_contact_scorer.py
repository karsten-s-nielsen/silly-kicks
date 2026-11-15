"""build_contact_scorer (scorer spec sections 3 and 4): the folds, the labels, the development run on the toy, the
held-out guards and the token that reads the frozen scorer's content."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.scripts.test_gs_truth import _DEV


def test_fold_assignment_is_six_by_four():
    import scripts.build_contact_scorer as d

    folds = d.fold_of(_DEV)
    assert sorted(folds) == sorted(_DEV)
    assert sorted(pd.Series(folds).value_counts().tolist()) == [4] * 6
    order = sorted(_DEV, key=lambda g: hashlib.sha256(g.encode()).hexdigest())
    assert [folds[g] for g in order] == [i % 6 for i in range(24)]  # position in sha256 order, mod 6


def _opts(rows):
    """rows: (period, t, player)."""
    return pd.DataFrame(
        {"period_id": [r[0] for r in rows], "time_seconds": [r[1] for r in rows], "player_id": [r[2] for r in rows]}
    )


_SEQ = pd.DataFrame(
    {"period_id": [1, 1], "t0": [0.0, 3.0], "t1": [2.0, 6.0], "player": ["7", "9"], "ptype": ["PA", "PA"]}
)


def test_labels_by_region():
    import scripts.build_contact_scorer as d

    lab = d.label_options(
        _opts(
            [
                (1, 2.05, 7),  # near 7's release (2.0): the edge's player -> 1
                (1, 2.05, 9),  # same peak, another option -> 0
                (1, 2.5, 9),  # in flight (2.0 .. 3.0, > 0.2 from both edges) -> 0
                (1, 4.5, 9),  # inside 9's sequence, away from its edges -> not used
                (1, 2.9, 9),  # near 9's reception (3.0) -> 1
            ]
        ),
        _SEQ,
    )
    assert lab[:3].tolist() == [1.0, 0.0, 0.0] and np.isnan(lab[3]) and lab[4] == 1.0


def test_labels_take_the_nearer_edge():
    import scripts.build_contact_scorer as d

    seq = pd.DataFrame(
        {"period_id": [1, 1], "t0": [0.0, 2.1], "t1": [2.0, 4.0], "player": ["7", "9"], "ptype": ["PA"] * 2}
    )
    lab = d.label_options(_opts([(1, 2.08, 9), (1, 2.02, 7)]), seq)  # 2.08 is nearer 2.1 (9); 2.02 nearer 2.0 (7)
    assert lab.tolist() == [1.0, 1.0]


def test_dev_run_writes_the_report_and_the_frozen_scorer(tmp_path):
    from scripts._toy_corpora import contact_scorer_toy
    from silly_kicks.tracking._contact_scorer import ContactScorer
    from tests.scripts._parity import run_serial

    data = contact_scorer_toy("toycontact_scorer")(tmp_path / "data")
    out = tmp_path / "out"
    run_serial(
        "scripts.build_contact_scorer", tmp_path / "s", data, ["--dataset", "toycontact_scorer", "--split", "dev"], out
    )
    rep = json.loads((out / "contact_scorer_dev.json").read_text(encoding="utf-8"))
    assert rep["split"] == "dev" and rep["matches"] == ["3816", "3819", "3820", "3822"]  # no held-out game read
    assert [g["threshold"] for g in rep["grid"]] == [round(0.05 * i, 2) for i in range(1, 20)]
    model = out / "model.json"
    assert hashlib.sha256(model.read_bytes()).hexdigest() == rep["scorer_model_sha256"]
    ContactScorer.load(out)  # the frozen scorer, beside the report, passes its own checksum and contract


def _ctx(tmp_path, *, commit="abc", tree="clean", dev=None, tamper=False):
    """A fake context for the held-out report: a development report with a frozen scorer beside it."""
    import scripts.build_contact_scorer as d

    rng = np.random.default_rng(0)
    from silly_kicks.tracking._contact_scorer import FEATURE_NAMES, ContactScorer

    x = pd.DataFrame({n: rng.normal(size=50) for n in FEATURE_NAMES})
    sc = ContactScorer().fit(x, (x["dist"] < 0).astype(int).to_numpy())
    sc.save(tmp_path)
    sha = d._sha(tmp_path / "model.json")
    rep = {
        "split": "dev",
        "run_commit": "abc",
        "run_tree_dirty": False,
        "chosen": {"min_contact_probability": 0.5},
        "scorer_model_sha256": sha,
        **(dev or {}),
    }
    (tmp_path / "dev.json").write_text(json.dumps(rep), encoding="utf-8")
    if tamper:
        m = tmp_path / "model.json"
        m.write_text(m.read_text(encoding="utf-8").replace('"baseline"', '"baseline" ', 1), encoding="utf-8")
    seqs = _SEQ.assign(game="3812", supported=True)
    opts = _opts([(1, 2.05, "7")]).assign(
        game="3812", label=np.nan, peak_id=0, rank=0.0, **{n: 0.0 for n in FEATURE_NAMES if n not in ("rank",)}
    )
    up = {"options": opts, "sequences": seqs}
    return SimpleNamespace(
        params={"split": "holdout", "dev_report": str(tmp_path / "dev.json"), "dataset": "gs_contact_truth"},
        prov={"commit": commit, "tree_state": tree},
        upstream=lambda name: up[name],
    )


def test_holdout_scores_the_frozen_scorer_once(tmp_path):
    import scripts.build_contact_scorer as d

    files = d._holdout_report(_ctx(tmp_path)).files
    rep = json.loads(files["contact_scorer_holdout.json"])
    assert rep["split"] == "holdout" and rep["matches"] == ["3812"] and rep["min_contact_probability"] == 0.5


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"commit": "zzz"}, "from commit abc, not zzz"),
        ({"tree": "dirty"}, "clean"),
        ({"dev": {"run_tree_dirty": True}}, "clean"),
        ({"dev": {"chosen": None}}, "feasible"),
        ({"dev": {"split": "holdout"}}, "feasible"),
        ({"tamper": True}, "differs from the scorer"),
    ],
)
def test_holdout_guards(tmp_path, kw, match):
    import scripts.build_contact_scorer as d

    with pytest.raises(ValueError, match=match):
        d._holdout_report(_ctx(tmp_path, **kw))


def test_holdout_needs_a_dev_report(tmp_path):
    import scripts.build_contact_scorer as d

    ctx = _ctx(tmp_path)
    ctx.params["dev_report"] = None
    with pytest.raises(ValueError, match="--dev-report"):
        d._holdout_report(ctx)


def test_a_refit_at_the_same_path_changes_the_token(tmp_path):
    import scripts.build_contact_scorer as d

    _ctx(tmp_path)
    params = {"split": "holdout", "dev_report": str(tmp_path / "dev.json")}
    before = d._declare_scorer(params)
    m = tmp_path / "model.json"
    m.write_text(m.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert d._declare_scorer(params) != before
    assert d._declare_scorer({"split": "dev", "dev_report": None}) == {}


# --------------------------------------------------------------------------- review A (scorer impl): the protocol
def _grid_with_best(best: float) -> dict:
    """A grid where only ``best`` and the next lower threshold are feasible and tied on "right"."""
    feasible = _m(0.70, 0.05)
    out = {}
    for th in [round(0.05 * i, 2) for i in range(1, 20)]:
        ok = th in (best, round(best - 0.05, 2))
        out[th] = feasible if ok else _m(0.30, 0.50)
    return out


def _m(right, false_first, sup_right=None, sup_false_first=None):
    """One grid row with both populations (scorer spec r7 section 3); the supported rows default to the all-pairs."""
    return {
        "right_share": right,
        "false_first_share": false_first,
        "sup_right_share": right if sup_right is None else sup_right,
        "sup_false_first_share": false_first if sup_false_first is None else sup_false_first,
        "timing_median_s": 0.0,
    }


@pytest.mark.parametrize(("best", "edge"), [(0.5, False), (0.95, True), (0.1, False)])
def test_selection_takes_the_larger_threshold_and_reports_the_edge(best, edge):
    import scripts.build_contact_scorer as d

    chosen, on_edge = d._select(_grid_with_best(best))
    assert chosen == best and on_edge is edge  # the larger of the two tied feasible thresholds
    assert d._select({0.05: _m(0.7, 0.0)}) == (0.05, True)
    assert d._select({0.5: _m(0.1, 0.9)}) == (None, False)


def test_fold_fits_never_see_the_held_fold(monkeypatch):
    import scripts.build_contact_scorer as d

    games = list(_DEV)
    opts = pd.DataFrame({"game": games, "label": [1.0] * len(games)})
    seqs = pd.DataFrame({"game": games})
    trained: list[set] = []
    monkeypatch.setattr(d, "_fit", lambda o: trained.append(set(o["game"])) or object())
    monkeypatch.setattr(d, "_score", lambda o, s, sc, th: (dict.fromkeys(d._COUNTS_ALL, 0), []))
    up = {"options": opts, "sequences": seqs}
    ctx = SimpleNamespace(params={"split": "dev"}, upstream=lambda n: up[n])
    folds = d.fold_of(games)
    for f in range(6):
        d._fold_work(str(f), ctx)
        held = {g for g, k in folds.items() if k == f}
        assert trained[-1] == set(games) - held and not (trained[-1] & held)


def test_refit_uses_every_development_match(monkeypatch):
    import scripts.build_contact_scorer as d

    games = ["3816", "3819", "3820"]
    opts = pd.DataFrame({"game": games, "label": [1.0, 0.0, 1.0]})
    seqs = pd.DataFrame({"game": games})
    oof = pd.DataFrame([{"threshold": th, **dict.fromkeys(d._COUNTS_ALL, 0), "release_offsets": "[]"} for th in d.GRID])
    seen: list[set] = []
    monkeypatch.setattr(d, "_fit", lambda o: seen.append(set(o["game"])) or object())
    monkeypatch.setattr(d, "_scorer_files", lambda sc, report: {"model.json": b"{}"})
    up = {"options": opts, "sequences": seqs, "folds": oof}
    ctx = SimpleNamespace(
        params={"split": "dev", "dataset": "gs_contact_truth"},
        prov={"commit": "abc", "tree_state": "clean"},
        upstream=lambda n: up[n],
    )
    d._dev_report(ctx)
    assert seen == [set(games)]


def test_holdout_scores_at_the_development_threshold(tmp_path, monkeypatch):
    import scripts.build_contact_scorer as d

    used = []
    real = d._score
    monkeypatch.setattr(d, "_score", lambda o, s, sc, th: used.append(th) or real(o, s, sc, th))
    rep = json.loads(
        d._holdout_report(_ctx(tmp_path, dev={"chosen": {"min_contact_probability": 0.35}})).files[
            "contact_scorer_holdout.json"
        ]
    )
    assert used == [0.35] and rep["min_contact_probability"] == 0.35


def test_dev_run_writes_metadata_and_metrics(tmp_path):
    from scripts._toy_corpora import contact_scorer_toy
    from tests.scripts._parity import run_serial

    data = contact_scorer_toy("toycontact_scorer")(tmp_path / "data")
    out = tmp_path / "out"
    run_serial(
        "scripts.build_contact_scorer", tmp_path / "s", data, ["--dataset", "toycontact_scorer", "--split", "dev"], out
    )
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["corpus_provenance"] == {
        "providers": ["gradientsports"],
        "n_games": 4,
        "game_ids": ["3816", "3819", "3820", "3822"],
    }
    assert meta["stores_training_data"] is False and meta["frames_dataset"] == "toycontact_scorer"
    rep = json.loads((out / "contact_scorer_dev.json").read_text(encoding="utf-8"))
    assert meta["training_commit"] == rep["run_commit"]  # the manifest commit_field reads this value
    assert meta["min_contact_probability"] == (rep["chosen"] or {}).get("min_contact_probability")
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert {"run_commit", "run_tree_dirty", "oof_at_chosen", "refit_dev_at_chosen"} <= set(metrics)
    chosen = (rep["chosen"] or {}).get("min_contact_probability")
    row = next((g for g in rep["grid"] if g["threshold"] == chosen), None)
    assert metrics["oof_at_chosen"] == row and metrics["run_commit"] == rep["run_commit"]
    sums = (out / "SHA256SUMS").read_text(encoding="utf-8")
    assert all(name in sums for name in ("model.json", "metadata.json", "metrics.json"))


def test_flight_labels_need_a_sequence_on_each_side():
    import scripts.build_contact_scorer as d

    seq = pd.DataFrame(
        {"period_id": [1, 1], "t0": [2.0, 6.0], "t1": [4.0, 8.0], "player": ["7", "9"], "ptype": ["PA"] * 2}
    )
    lab = d.label_options(_opts([(1, 1.0, 7), (1, 5.0, 7), (1, 9.0, 9)]), seq)
    assert np.isnan(lab[0]) and lab[1] == 0.0 and np.isnan(lab[2])  # before the first / after the last: not used


# --------------------------------------------------------------------------- review B (scorer impl): isolation
def test_each_mode_reads_only_its_own_half(tmp_path):
    import scripts.build_contact_scorer as d
    from scripts._toy_corpora import contact_scorer_toy

    data = contact_scorer_toy("toycontact_scorer")(tmp_path / "data")

    def refs(split):
        return d._refs(SimpleNamespace(params={"split": split, "dataset": "toycontact_scorer"}, data_root=data))

    assert sorted(d._game(r) for r in refs("dev")) == ["3816", "3819", "3820", "3822"]
    assert sorted(d._game(r) for r in refs("holdout")) == ["3812"]
    assert all(f.endswith(("/frames.parquet", "/sequences.parquet")) for r in refs("dev") for f in r.files)


def test_holdout_mode_runs_no_folds():
    import scripts.build_contact_scorer as d

    assert d._fold_units(SimpleNamespace(params={"split": "holdout"})) == []
    assert d._fold_units(SimpleNamespace(params={"split": "dev"})) == [str(f) for f in range(6)]


def test_reports_carry_the_dataset_revision(tmp_path, monkeypatch):
    import scripts.build_contact_scorer as d
    from scripts import _driver_sources

    monkeypatch.setattr(_driver_sources, "_lock_units", lambda ds: {"revision": "rev123"} if ds else None)
    rep = json.loads(d._holdout_report(_ctx(tmp_path)).files["contact_scorer_holdout.json"])
    assert rep["dataset_revision"] == "rev123" and rep["dataset"] == "gs_contact_truth"


def test_the_dev_report_carries_the_dataset_revision(monkeypatch):
    import scripts.build_contact_scorer as d
    from scripts import _driver_sources

    games = ["3816", "3819"]
    oof = pd.DataFrame([{"threshold": th, **dict.fromkeys(d._COUNTS_ALL, 0), "release_offsets": "[]"} for th in d.GRID])
    up = {
        "options": pd.DataFrame({"game": games, "label": [1.0, 0.0]}),
        "sequences": pd.DataFrame({"game": games}),
        "folds": oof,
    }
    monkeypatch.setattr(_driver_sources, "_lock_units", lambda ds: {"revision": "rev456"} if ds else None)
    monkeypatch.setattr(d, "_fit", lambda o: object())
    monkeypatch.setattr(d, "_scorer_files", lambda sc, report: {"model.json": b"{}"})
    ctx = SimpleNamespace(
        params={"split": "dev", "dataset": "gs_contact_truth"},
        prov={"commit": "abc", "tree_state": "clean"},
        upstream=lambda n: up[n],
    )
    rep = json.loads(d._dev_report(ctx).files["contact_scorer_dev.json"])
    assert rep["dataset_revision"] == "rev456" and rep["dataset"] == "gs_contact_truth"


# --------------------------------------------------------------------------- scorer spec r7: the v2 driver
def test_the_sequences_stage_adds_the_support_flag():
    import scripts.build_contact_scorer as d
    from tests.tracking.test_ball_contact import _scene

    frames = _scene([(0.0, 4.0, (50.0, 34.0), (50.0, 34.0))], {7: ("T1", 50.5, 34.0), 9: ("T2", 60.0, 34.0)})
    out = d._sequences_work(("3816", frames, _SEQ), SimpleNamespace(params={}))
    assert out["supported"].dtype == bool and out["supported"].tolist() == [True, False]  # 9 is 10 m away
    assert out["game"].tolist() == ["3816", "3816"] and out["player"].tolist() == ["7", "9"]


def test_the_sequences_stage_loads_the_frames():
    from typing import Any, cast

    import scripts.build_contact_scorer as d
    from scripts._stages import ItemStage

    stage = cast("ItemStage", next(s for s in d.PIPELINE.stages if s.name == "sequences"))
    load = stage.load
    assert load is not None
    calls = []
    original = d._read
    try:
        d._read = lambda r, c, name: calls.append(name) or pd.DataFrame()
        load(SimpleNamespace(stem="gradientsports-3816"), cast("Any", SimpleNamespace()))
    finally:
        d._read = original
    assert calls == ["frames.parquet", "sequences.parquet"]


def test_the_selection_reads_the_supported_rows():
    import scripts.build_contact_scorer as d

    grid = {0.4: _m(0.62, 0.09, sup_right=0.50, sup_false_first=0.12), 0.5: _m(0.45, 0.20, 0.66, 0.09)}
    assert d._select(grid) == (0.5, False)


def test_the_score_decodes_with_the_scorers_window(monkeypatch):
    import scripts.build_contact_scorer as d
    from silly_kicks.tracking import _contact_scorer as cs

    seen = []
    real = cs.credit_options

    def spy(opts, prob, params, *, decode_window_s):
        seen.append(decode_window_s)
        return real(opts, prob, params, decode_window_s=decode_window_s)

    monkeypatch.setattr(cs, "credit_options", spy)
    scorer = SimpleNamespace(decode_window_s=0.27, predict_proba=lambda o: np.full(len(o), 0.9))
    opts = _opts([(1, 3.05, "9")]).assign(game="3816", peak_id=0, rank=0.0, z=np.nan)
    counts, _ = d._score(opts, _SEQ.assign(game="3816", supported=True), scorer, 0.5)
    assert seen == [0.27]
    assert (counts["n"], counts["right"], counts["sup_n"], counts["sup_right"]) == (1, 1, 1, 1)


def test_the_reports_carry_both_populations(tmp_path):
    import sklearn

    from scripts._toy_corpora import contact_scorer_toy
    from tests.scripts._parity import run_serial

    data = contact_scorer_toy("toycontact_scorer")(tmp_path / "data")
    out = tmp_path / "out"
    run_serial(
        "scripts.build_contact_scorer", tmp_path / "s", data, ["--dataset", "toycontact_scorer", "--split", "dev"], out
    )
    rep = json.loads((out / "contact_scorer_dev.json").read_text(encoding="utf-8"))
    keys = {"right_share", "false_first_share", "sup_n", "sup_right_share", "sup_false_first_share", "supported_share"}
    assert all(keys <= set(g) for g in rep["grid"])
    assert rep["bar_population"] == "supported"
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["sklearn_version"] == sklearn.__version__


def test_the_holdout_report_judges_the_supported_pairs(tmp_path, monkeypatch):
    import scripts.build_contact_scorer as d

    seen = []
    real = d.meets_bar
    monkeypatch.setattr(
        d, "meets_bar", lambda m, *, population: seen.append(population) or real(m, population=population)
    )
    rep = json.loads(d._holdout_report(_ctx(tmp_path)).files["contact_scorer_holdout.json"])
    assert {"sup_n", "sup_right_share", "supported_share"} <= set(rep) and rep["bar_population"] == "supported"
    assert seen == ["supported"]
