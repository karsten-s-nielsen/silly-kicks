"""Unit tests for silly_kicks.tracking.kloppy gateway."""

import pytest
from datasets.tracking.metrica.generate_synthetic import (
    build_metrica_tracking_dataset,
)
from datasets.tracking.skillcorner.generate_synthetic import (
    build_skillcorner_tracking_dataset,
)

from silly_kicks.tracking.kloppy import convert_to_frames
from silly_kicks.tracking.schema import (
    KLOPPY_TRACKING_FRAMES_COLUMNS,
    TrackingConversionReport,
)


def test_metrica_dataset_converts_to_canonical_schema():
    ds = build_metrica_tracking_dataset(n_frames=10)
    frames, report = convert_to_frames(ds)
    assert set(frames.columns) == set(KLOPPY_TRACKING_FRAMES_COLUMNS)
    assert isinstance(report, TrackingConversionReport)
    assert report.provider == "metrica"


def test_metrica_coordinates_in_spadl_units():
    ds = build_metrica_tracking_dataset(n_frames=10)
    frames, _ = convert_to_frames(ds)
    assert frames["x"].between(0, 105).all()
    assert frames["y"].between(0, 68).all()


def test_metrica_derived_speed_used_when_native_missing():
    """Metrica's kloppy data may lack speed; gateway should derive."""
    ds = build_metrica_tracking_dataset(n_frames=10)
    for frame in ds.records:
        for pdata in frame.players_data.values():
            pdata.speed = None
        frame.ball_speed = None
    frames, report = convert_to_frames(ds)
    assert (frames["speed_source"] == "derived").any() or report.derived_speed_rows > 0


def test_skillcorner_dataset_converts():
    ds = build_skillcorner_tracking_dataset(n_frames=10)
    frames, report = convert_to_frames(ds)
    assert set(frames.columns) == set(KLOPPY_TRACKING_FRAMES_COLUMNS)
    assert report.provider == "skillcorner"


def test_provider_pff_raises_not_implemented():
    """Per ADR-004 invariant 4: route PFF through silly_kicks.tracking.pff."""
    from kloppy.domain import Provider

    ds = build_metrica_tracking_dataset(n_frames=2)
    ds.metadata.provider = Provider.PFF
    with pytest.raises(NotImplementedError, match=r"silly_kicks\.tracking\.pff"):
        convert_to_frames(ds)


def test_provider_sportec_raises_not_implemented():
    from kloppy.domain import Provider

    ds = build_metrica_tracking_dataset(n_frames=2)
    ds.metadata.provider = Provider.SPORTEC
    with pytest.raises(NotImplementedError, match=r"silly_kicks\.tracking\.sportec"):
        convert_to_frames(ds)
