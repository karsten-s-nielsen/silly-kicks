"""diagnose_provider ``coords`` aspect: shape, JSON-safety, thin-wrapper equality, real-data negative, bad aspect."""

import json

import pytest

from silly_kicks.mcp import server


def test_coords_aspect_shape(fixture_ref):
    r = server.diagnose_provider("idsse", fixture_ref, "coords")
    assert r["aspect"] == "coords"
    assert set(r) >= {"provider", "match_ref", "aspect", "findings", "flags"}


def test_coords_findings_json_serializable(fixture_ref):  # nested CoordinateDiagnosis -> asdict
    json.dumps(server.diagnose_provider("idsse", fixture_ref, "coords")["findings"])


def test_coords_thin_wrapper_equality(fixture_ref, _server_load):
    from silly_kicks.spadl import diagnose_coordinates

    loaded = _server_load["OK"]
    diag = diagnose_coordinates(loaded.actions, loaded.frames)
    r = server.diagnose_provider("idsse", fixture_ref, "coords")
    assert r["findings"] == server._json_safe(diag)  # adapter adds no analysis
    assert r["flags"] == list(diag.flags)


def test_coords_real_slice_no_scale_defect(fixture_ref):
    # the real elastic_sync slice is genuine SPADL meters -> must never trip the scale/units tripwire
    r = server.diagnose_provider("idsse", fixture_ref, "coords")
    assert "coords_scale_suspect" not in r["flags"]


def test_unknown_aspect_still_raises(fixture_ref):
    with pytest.raises(ValueError):
        server.diagnose_provider("idsse", fixture_ref, "bogus")
