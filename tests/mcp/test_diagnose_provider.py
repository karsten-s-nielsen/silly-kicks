"""diagnose_provider: per-aspect shape, JSON-safety (all aspects), thin-wrapper, fail-loud, read-only."""

import json

import pytest

from silly_kicks.mcp import server

ASPECTS = ["keeper", "convention", "id_dtype"]


@pytest.mark.parametrize("aspect", ASPECTS)
def test_aspect_shape(fixture_ref, aspect):
    r = server.diagnose_provider("idsse", fixture_ref, aspect)
    assert set(r) >= {"provider", "match_ref", "aspect", "findings", "flags"}


@pytest.mark.parametrize("aspect", ASPECTS)
def test_findings_json_serializable(fixture_ref, aspect):  # tuple keys + numpy scalars
    json.dumps(server.diagnose_provider("idsse", fixture_ref, aspect)["findings"])


def test_thin_wrapper_equality_id_dtype(fixture_ref, _server_load):
    from silly_kicks.tracking import validate_id_dtypes

    loaded = _server_load["OK"]
    diag = validate_id_dtypes(loaded.actions, loaded.frames, on_mismatch="warn")
    assert server.diagnose_provider("idsse", fixture_ref, "id_dtype")["findings"] == server._json_safe(diag)


def test_fail_loud_on_bad_load(_server_load):
    with pytest.raises(RuntimeError):
        server.diagnose_provider("idsse", "NOT_IN_REGISTRY", "keeper")


def test_read_only_on_dirty_tree(dirty_tree, fixture_ref):
    server.diagnose_provider("idsse", fixture_ref, "convention")
    assert dirty_tree.no_new_artifacts()
