"""TF-46 Task 3: the per-ROW stoppage-source transport on StudyInputs (spec section 27.19).

The trainer stamps the shipped bundle's stoppage provenance from exactly the shipped-arm rows, so the
per-row source must ride the persisted study inputs (``len == len(X)``, like ``match_ids``/``providers``)
and round-trip byte-for-byte. A pre-TF-46 inputs dir (no ``stoppage_sources.npy``) loads an EMPTY array,
whose combine FAILS LOUD -- never a synthesized ``ball_state`` (SPEC-20)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts._study_shared import load_study_inputs, persist_study_inputs, provenance_from_sources

_EVENTS_V3 = "events_v3"  # the current STOPPAGE_SOURCES token for an event-only provider


def _persist(tmp_path, x, *, stoppage_sources):
    persist_study_inputs(
        tmp_path,
        X=x,
        y=np.zeros(len(x), dtype=int),
        groups=np.array([f"g{i}" for i in range(len(x))]),
        providers=np.array(["skillcorner"] * len(x)),
        match_ids=np.array([str(i) for i in range(len(x))]),
        is_public=np.ones(len(x), dtype=bool),
        config={},
        stoppage_sources=stoppage_sources,
    )


def test_persist_load_round_trip_carries_sources(tmp_path):
    X = pd.DataFrame({"f": [0.0, 1.0, 2.0]})
    sources = np.array([_EVENTS_V3, _EVENTS_V3, "ball_state"], dtype=object)
    _persist(tmp_path, X, stoppage_sources=sources)
    inp = load_study_inputs(tmp_path)
    assert list(inp.stoppage_sources) == [_EVENTS_V3, _EVENTS_V3, "ball_state"]
    assert len(inp.stoppage_sources) == len(inp.X)  # per-ROW alignment


def test_length_mismatch_raises(tmp_path):
    X = pd.DataFrame({"f": [0.0, 1.0, 2.0]})
    with pytest.raises(ValueError, match="per-ROW"):
        _persist(tmp_path, X, stoppage_sources=np.array([_EVENTS_V3, _EVENTS_V3], dtype=object))


def test_absent_sources_file_tolerated(tmp_path):
    # a pre-TF-46 inputs dir (written with stoppage_sources=None) loads an EMPTY array, not an error
    X = pd.DataFrame({"f": [0.0, 1.0]})
    _persist(tmp_path, X, stoppage_sources=None)
    inp = load_study_inputs(tmp_path)
    assert isinstance(inp.stoppage_sources, np.ndarray)
    assert len(inp.stoppage_sources) == 0


def test_default_yields_no_parts_not_ball_state():
    # SPEC-20: the EMPTY default combines to NO parts and RAISES -- never a synthesized ["ball_state"]
    from scripts._study_shared import StudyInputs

    empty = StudyInputs(
        X=pd.DataFrame({"f": [0.0, 1.0]}),
        y=np.zeros(2, int),
        groups=np.array(["g0", "g1"]),
        providers=np.array(["skillcorner", "skillcorner"]),
        match_ids=np.array(["0", "1"]),
        is_public=np.ones(2, bool),
        config={},
    ).stoppage_sources
    assert len(empty) == 0  # the field default is empty (absent inputs dir)
    with pytest.raises(ValueError, match="no parts"):
        provenance_from_sources(empty, np.array(["0", "1"]), np.array([True, True]))


def test_provenance_from_sources_unions_shipped_rows():
    # non-vacuity: two matches with different sources -> the union, one part per match, validated
    sources = np.array([_EVENTS_V3, _EVENTS_V3, "ball_state"], dtype=object)
    match_ids = np.array(["m1", "m1", "m2"])
    mask = np.array([True, True, True])
    prov = provenance_from_sources(sources, match_ids, mask)
    assert prov["stoppage_sources"] == ["ball_state", _EVENTS_V3]  # sorted union across both matches
    assert prov["stoppage_rule_version"] == 3
    # the shipped-arm mask narrows it: drop the ball_state match -> only events_v3 remains
    only_m1 = provenance_from_sources(sources, match_ids, np.array([True, True, False]))
    assert only_m1["stoppage_sources"] == [_EVENTS_V3]
