"""A reader-built model records its stoppage provenance; load() refuses one without it (TF-46 Part 2 spec 27.19).

Owner decision 2026-10-26 (models): refuse, retrain first. ``save()`` refuses a model without a current provenance,
``load()`` refuses an artifact without one (``IntegrityError``), and ``legacy_override=True`` loads it with a
``StoppageProvenanceWarning`` (audit use only). The structural guards (SHA, pitch, chirality, feature contract) run
first, so a tampered artifact still fails on its own guard.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from silly_kicks.tracking import (
    STOPPAGE_RULE_VERSION,
    STOPPAGE_SOURCES,
    StoppageProvenanceWarning,
    combine_stoppage_provenance,
    stoppage_provenance,
    stoppage_provenance_problem,
)
from silly_kicks.tracking import _xcross_attempt as xc
from silly_kicks.tracking import _xshot_occurrence as xs
from silly_kicks.tracking._xshot_occurrence import IntegrityError
from tests._stoppage_fixtures import toy_stoppage_provenance, with_toy_stoppage_provenance


def _xs_model():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 27)), columns=xs.XSHOT_FEATURE_NAMES_FAITHFUL)
    y = (X["r"] + rng.normal(scale=0.5, size=200) < 0).astype(int)
    return xs.XShotOccurrenceModel().fit(X, pd.Series(y)), X


def _xc_model():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 16)), columns=xc.XCROSS_FEATURE_NAMES_FAITHFUL)
    y = (X["gk_r"] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    return xc.XCrossAttemptModel().fit(X, pd.Series(y)), X


MODELS = [
    pytest.param(_xs_model, xs.XShotOccurrenceModel, id="xshot"),
    pytest.param(_xc_model, xc.XCrossAttemptModel, id="xcross"),
]


def _rewrite_meta(d, edit) -> None:
    """Edit metadata.json and re-sum it, so the SHA guard passes and the stoppage guard is the one that decides."""
    meta = json.loads((d / "metadata.json").read_text())
    edit(meta)
    (d / "metadata.json").write_text(json.dumps(meta, indent=2), newline="\n")
    with open(d / "SHA256SUMS", "w", newline="\n") as f:
        for fn in ["model.json", "metadata.json"]:
            raw = (d / fn).read_bytes().replace(b"\r\n", b"\n")
            f.write(f"{hashlib.sha256(raw).hexdigest()}  {fn}\n")


@pytest.mark.parametrize(("make", "cls"), MODELS)
def test_save_records_and_load_restores_the_provenance(tmp_path, make, cls):
    m, X = make()
    with_toy_stoppage_provenance(m).save(tmp_path / "m")
    meta = json.loads((tmp_path / "m" / "metadata.json").read_text())
    assert meta["stoppage_rule_version"] == STOPPAGE_RULE_VERSION
    assert meta["stoppage_sources"] == toy_stoppage_provenance()["stoppage_sources"]
    assert list(meta)[-1] == "training_commit"  # the bundled key order stays: training_commit LAST
    back = cls.load(tmp_path / "m")
    assert back.stoppage_provenance == toy_stoppage_provenance()
    np.testing.assert_allclose(back.predict_proba(X), m.predict_proba(X), rtol=1e-9)


@pytest.mark.parametrize(("make", "cls"), MODELS)
def test_save_refuses_a_model_without_a_provenance(tmp_path, make, cls):
    m, _ = make()
    with pytest.raises(RuntimeError, match="stoppage provenance"):
        m.save(tmp_path / "m")
    assert not (tmp_path / "m" / "metadata.json").exists()  # nothing half-written
    m.stoppage_provenance = {"stoppage_rule_version": STOPPAGE_RULE_VERSION, "stoppage_sources": ["ball_state_v0"]}
    with pytest.raises(RuntimeError, match="not current"):
        m.save(tmp_path / "m")


@pytest.mark.parametrize(("make", "cls"), MODELS)
@pytest.mark.parametrize(
    ("edit", "match"),
    [
        pytest.param(
            lambda m: [m.pop("stoppage_rule_version"), m.pop("stoppage_sources")], "no stoppage provenance", id="absent"
        ),
        pytest.param(lambda m: m.pop("stoppage_sources"), "no stoppage provenance", id="half"),
        pytest.param(lambda m: m.update(stoppage_rule_version=STOPPAGE_RULE_VERSION + 1), "rule version", id="version"),
        pytest.param(lambda m: m.update(stoppage_sources=["events_v0"]), "not current", id="source"),
        pytest.param(lambda m: m.update(stoppage_sources=[]), "non-empty", id="empty"),
    ],
)
def test_load_refuses_an_artifact_without_a_current_provenance(tmp_path, make, cls, edit, match):
    m, _ = make()
    with_toy_stoppage_provenance(m).save(tmp_path / "m")
    cls.load(tmp_path / "m")  # precondition: the unedited artifact loads
    _rewrite_meta(tmp_path / "m", edit)
    with pytest.raises(IntegrityError, match=match):
        cls.load(tmp_path / "m")


@pytest.mark.parametrize(("make", "cls"), MODELS)
def test_legacy_override_loads_a_pre_column_artifact_with_a_warning(tmp_path, make, cls):
    m, X = make()
    with_toy_stoppage_provenance(m).save(tmp_path / "m")
    _rewrite_meta(tmp_path / "m", lambda meta: [meta.pop("stoppage_rule_version"), meta.pop("stoppage_sources")])
    with pytest.warns(StoppageProvenanceWarning, match="audit use only"):
        back = cls.load(tmp_path / "m", legacy_override=True)
    assert back.stoppage_provenance is None  # never back-filled
    np.testing.assert_allclose(back.predict_proba(X), m.predict_proba(X), rtol=1e-9)


@pytest.mark.parametrize(("make", "cls"), MODELS)
def test_a_structural_guard_still_fires_before_the_stoppage_guard(tmp_path, make, cls):
    """Order: the chirality guard decides a tampered artifact that also lacks the provenance."""
    m, _ = make()
    with_toy_stoppage_provenance(m).save(tmp_path / "m")

    def _both(meta):
        meta["chirality"]["outputs"] = [v + 0.5 for v in meta["chirality"]["outputs"]]
        meta.pop("stoppage_rule_version")

    _rewrite_meta(tmp_path / "m", _both)
    with pytest.raises(IntegrityError, match="chirality"):
        cls.load(tmp_path / "m")


def test_stoppage_provenance_reads_the_frames():
    live = pd.DataFrame(
        {
            "source_provider": ["skillcorner", "skillcorner", "idsse"],
            "stoppage_state": ["live", "dead", "live"],
            "stoppage_source": [STOPPAGE_SOURCES[1], STOPPAGE_SOURCES[1], "ball_state"],
        }
    )
    assert stoppage_provenance(live, where="t") == {
        "stoppage_rule_version": STOPPAGE_RULE_VERSION,
        "stoppage_sources": sorted(["ball_state", STOPPAGE_SOURCES[1]]),
    }
    # an observed provider without the column keeps ball_state: the same mapping
    observed = pd.DataFrame({"source_provider": ["sportec"], "ball_state": ["alive"]})
    assert stoppage_provenance(observed, where="t")["stoppage_sources"] == ["ball_state"]
    # the reader gate runs first: SkillCorner without the column raises where a reader would
    with pytest.raises(ValueError, match="no observed dead-ball state"):
        stoppage_provenance(pd.DataFrame({"source_provider": ["skillcorner"], "ball_state": ["alive"]}), where="t")
    assert stoppage_provenance(observed.iloc[:0], where="t")["stoppage_sources"] == []
    assert stoppage_provenance_problem(stoppage_provenance(observed.iloc[:0], where="t")) is not None


def test_combine_refuses_a_stale_part():
    good = toy_stoppage_provenance()
    with pytest.raises(ValueError, match="rule version"):
        combine_stoppage_provenance([good, {**good, "stoppage_rule_version": 0}])
    with pytest.raises(ValueError, match="no parts"):
        combine_stoppage_provenance([])
