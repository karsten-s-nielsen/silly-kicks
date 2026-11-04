"""TF-46 Task 4: the trainer stamps the shipped bundle's stoppage provenance from exactly the SHIPPED-ARM
rows (provenance-STAMPED ONLY this cycle -- no section 18.2.1 censoring, owner decision 2026-10-31), and
metrics.json carries the per-provider POST-DROP contributing-unit count.

The heavy end-to-end cases run the REAL stage pipeline on the staged toy corpus (``scripts/_toy_corpora``)
and are ``@pytest.mark.slow`` (the dedicated slow job, ADR-023). The wiring + derivation cases are cheap."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("ruthless")

from scripts import train_xcross_attempt as xc
from scripts import train_xshot_occurrence as xs
from scripts._study_shared import contributing_units_by_provider
from silly_kicks.tracking import STOPPAGE_RULE_VERSION, STOPPAGE_SOURCES
from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL, XCrossAttemptModel
from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL, XShotOccurrenceModel

_EVENTS_V3 = STOPPAGE_SOURCES[1]
TRAINERS = pytest.mark.parametrize("trainer", [xs, xc], ids=["xshot", "xcross"])


def _toy(tmp_path, trainer):
    """The trainer's staged toy dataset(s) under ``<tmp>/data``; returns ``(data_root, base_args)``."""
    from scripts._toy_corpora import build_xcross_corpus, build_xcross_probe_corpus, build_xshot_corpus

    data = tmp_path / "data"
    if trainer is xs:
        build_xshot_corpus(data / "toy", heavy_episodes=None)
        return data, ["--dataset", "toy"]
    build_xcross_corpus(data / "toy", heavy_episodes=None)
    build_xcross_probe_corpus(data / "probe")
    return data, ["--dataset", "toy", "--probe-dataset", "probe"]


# ---- cheap: derivation + model-level guard + static not-censored ----


def test_contributing_units_is_shipped_arm_not_full_roster():
    # the ACTUAL count is DISTINCT matches per provider over the SHIPPED mask, never the full roster (A PLAN-03)
    providers = np.array(["skillcorner", "skillcorner", "skillcorner", "gradientsports"])
    match_ids = np.array(["m1", "m1", "m2", "m9"])  # sc: 2 distinct matches; gs: 1
    shipped = np.array([True, True, True, False])  # the gradientsports row is NOT in the shipped arm
    counts = contributing_units_by_provider(providers, match_ids, shipped)
    assert counts == {"skillcorner": 2}  # per-provider distinct matches, shipped rows only (not {sc:2, gs:1})


def _tiny_fit(model_cls, feature_names):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, len(feature_names))), columns=feature_names)
    y = pd.Series((X.iloc[:, 0] + rng.normal(scale=0.5, size=120) < 0).astype(int))
    return model_cls().fit(X, y)


def test_save_raises_when_provenance_unset_xshot(tmp_path):
    # regression guard (B-10): save() refuses a model whose stoppage provenance is unset (spec section 27.19)
    m = _tiny_fit(XShotOccurrenceModel, XSHOT_FEATURE_NAMES_FAITHFUL)
    m.stoppage_provenance = None
    with pytest.raises(RuntimeError, match="stoppage"):
        m.save(tmp_path / "xs_v1")


def test_save_raises_when_provenance_unset_xcross(tmp_path):
    m = _tiny_fit(XCrossAttemptModel, XCROSS_FEATURE_NAMES_FAITHFUL)
    m.stoppage_provenance = None
    with pytest.raises(RuntimeError, match="stoppage"):
        m.save(tmp_path / "xc_v1")


def test_trainers_stamp_and_do_not_censor():
    # Provenance-STAMPED ONLY this cycle (owner 2026-10-31, plan B-40). An empty RESTART_GATE_TABLE makes every
    # derived restart unvalidated-time, so the section 18.2.1 censoring would strip every derived row; the
    # library therefore defaults it OFF until the joint release fills the table. The behavioural guard is
    # test_trainer_extract_keeps_rows_the_censor_would_drop; this one pins the default and the stamp.
    import inspect

    from silly_kicks.tracking._restart_validation import RESTART_GATE_TABLE
    from silly_kicks.tracking._xcross_attempt import prepare_xcross_training_data
    from silly_kicks.tracking._xshot_occurrence import prepare_xshot_training_data

    assert len(RESTART_GATE_TABLE) == 0, "RESTART_GATE_TABLE is expected empty until the joint release"
    for mod in (xs, xc):
        mod_file = mod.__file__
        assert mod_file is not None
        src = pathlib.Path(mod_file).read_text(encoding="utf-8")
        assert "provenance_from_sources" in src, f"{mod.__name__}: does not stamp the stoppage provenance"
    for fn in (prepare_xshot_training_data, prepare_xcross_training_data):
        assert inspect.signature(fn).parameters["apply_provenance_censor"].default is False, (
            f"{fn.__name__}: section 18.2.1 censoring must stay OFF by default until the joint release"
        )


def _extract_prep(trainer, tmp_path):
    """Run the trainer's REAL extract + prep stages (no HPO) on its toy corpus; return the persisted StudyInputs."""
    from scripts._stages import ItemStage, LocalStore, make_context, reduce_stage, run_unit, sorted_units
    from scripts._study_shared import load_study_inputs
    from scripts._toy_corpora import xshot_arms
    from tests.scripts._stage_harness import PROV, params_from_args

    data, base = _toy(tmp_path, trainer)
    args = [*base, "--arms-json", xshot_arms(data / "toy", paired=False), "--n-trials", "1"]
    ctx = make_context(
        trainer.PIPELINE,
        params_from_args(trainer.PIPELINE, args),
        PROV,
        store=LocalStore(tmp_path / "s"),
        data_root=data,
    )
    for name in ("extract", "prep"):
        st = trainer.PIPELINE.stage(name)
        refs = st.ref_map(ctx) if isinstance(st, ItemStage) else {}
        for key in sorted_units(ctx, st):
            run_unit(ctx, st, key, worker="test", ref=refs.get(key))
        reduce_stage(ctx, name)
    root, _inp = trainer._inputs(ctx)
    return load_study_inputs(root)


@TRAINERS
def test_trainer_extract_keeps_rows_the_censor_would_drop(trainer, tmp_path, monkeypatch):
    # B-40 behavioural guard (r6 B23): make the section 18.2.1 censor mark EVERY frame. A trainer (or a library
    # default) that applied the censoring would lose rows; the provenance-only refit must keep all of them.
    import silly_kicks.tracking._xcross_attempt as xca
    import silly_kicks.tracking._xshot_occurrence as xso

    baseline = len(_extract_prep(trainer, tmp_path / "base").X)
    assert baseline > 0  # non-vacuous: the toy produces training rows

    mod, name = (xso, "build_xshot_labels") if trainer is xs else (xca, "build_xcross_labels")
    real = getattr(mod, name)

    def censor_everything(*a, return_censor=False, **kw):
        if not return_censor:
            return real(*a, **kw)
        labels, censor = real(*a, return_censor=True, **kw)
        return labels, np.ones(len(np.asarray(censor)), dtype=bool)

    monkeypatch.setattr(mod, name, censor_everything)
    assert len(_extract_prep(trainer, tmp_path / "forced").X) == baseline


# ---- wiring: extract -> prep carries the per-ROW stoppage source onto StudyInputs ----


@TRAINERS
def test_prep_persists_per_row_stoppage_sources(trainer, tmp_path):
    """extract + prep only (no HPO): the toy's event-only frames carry ``events_v3``, so StudyInputs.stoppage_sources
    is per-ROW (``len == len(X)``) and all ``events_v3``."""
    loaded = _extract_prep(trainer, tmp_path)
    assert len(loaded.stoppage_sources) == len(loaded.X)  # per-ROW alignment
    assert set(map(str, loaded.stoppage_sources.tolist())) == {_EVENTS_V3}  # the toy is event-only


# ---- end-to-end: REAL prep -> fit stamps events_v3, load passes, metrics carries the count ----


@pytest.mark.slow  # slow: runs the full HPO pipeline on the toy (ADR-023), in the dedicated slow job
@TRAINERS
def test_fit_through_prep_stamps_events_v3(trainer, tmp_path):
    from scripts._toy_corpora import xshot_arms
    from tests.scripts._stage_harness import export, run_in_process

    model_cls = XShotOccurrenceModel if trainer is xs else XCrossAttemptModel
    data, base = _toy(tmp_path, trainer)
    args = [*base, "--arms-json", xshot_arms(data / "toy", paired=False), "--n-trials", "2"]
    ctx = run_in_process(trainer.PIPELINE, args, store=tmp_path / "store", data_root=data)
    out = export(ctx, tmp_path / "out")

    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["stoppage_rule_version"] == STOPPAGE_RULE_VERSION
    assert _EVENTS_V3 in meta["stoppage_sources"]  # stamped from the real (event-only) training rows
    model_cls.load(out)  # not-censored, valid provenance: load()/verify_stoppage_provenance passes (no IntegrityError)

    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    counts = metrics["contributing_units"]
    assert counts and all(int(v) >= 1 for v in counts.values())  # per-provider POST-DROP, non-empty
    # the full roster: exact ids for an all-public corpus, else the digest's n (corpus_identity)
    roster = metrics.get("corpus_n_matches") or len(metrics.get("corpus_match_ids", []))
    assert sum(int(v) for v in counts.values()) <= roster  # post-drop contributing <= the full roster
