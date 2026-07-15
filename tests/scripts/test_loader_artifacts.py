"""Loader must resolve BOTH SkillCorner artifact schemas (spec 3.1)."""

import pytest
from _loader_pining import _artifact_key, _dest_name

CANONICAL = {
    "1886347_dynamic_events": "1886347_dynamic_events.csv",
    "1886347_match": "1886347_match.json",
    "1886347_tracking_extrapolated": "1886347_tracking_extrapolated.jsonl",
}
ROLE_KEYED = {
    "events": "events.parquet",
    "freeze_frames": "freeze_frames.parquet",
    "metadata": "metadata.json",
    "physical": "physical.parquet",
    "tracking": "tracking.json.gz",
}


def test_suffix_resolution_still_works():
    assert _artifact_key(CANONICAL, suffix="_match.json", role="metadata") == "1886347_match"


def test_role_fallback_resolves_the_new_schema():
    assert _artifact_key(ROLE_KEYED, suffix="_match.json", role="metadata") == "metadata"


def test_unknown_role_and_suffix_raises():
    with pytest.raises(KeyError):
        _artifact_key(ROLE_KEYED, suffix="_nope.json", role="nonexistent")


def test_dest_name_preserves_the_extension():
    """kloppy sniffs the first byte: a gzip magic 0x1f under an extensionless name raises
    DeserializationError. The manifest's FILENAME must reach the temp file."""
    assert _dest_name("skillcorner", "1021404", "tracking", "tracking.json.gz").endswith(".json.gz")
    assert _dest_name("skillcorner", "1886347", "1886347_match", "1886347_match.json").endswith(".json")


def test_dest_name_is_stable_for_idsse_and_gs():
    """The rename must not break the providers that were already working."""
    assert _dest_name("idsse", "DFL-MAT-J03WMX", "tracking", "tracking.xml").endswith(".xml")
    assert _dest_name("gradientsports", "10502", "tracking", "tracking.jsonl.bz2").endswith(".jsonl.bz2")


def test_match_visibility_reads_the_manifest_field(monkeypatch):
    """The manifest already carries visibility: public | private (spec 3.2)."""
    import _loader_pining as lp

    monkeypatch.setattr(
        lp,
        "_list_matches",
        lambda provider, token, base_url: [
            {"id": "1886347", "visibility": "public", "artifacts": {}},
            {"id": "1021404", "visibility": "private", "artifacts": {}},
            {"id": "9999999", "artifacts": {}},  # field ABSENT
        ],
    )
    vis = lp.match_visibility(["skillcorner"], token="t", base_url="b")
    assert vis[("skillcorner", "1886347")] == "public"
    assert vis[("skillcorner", "1021404")] == "private"
    assert vis[("skillcorner", "9999999")] == "private"  # FAIL-CLOSED on an absent field
