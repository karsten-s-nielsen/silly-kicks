"""Offline-manifest mode for the pining loader: enumeration from a STAGED manifest, no token/network.

This is what lets an air-gapped run (e.g. an HF Job over a staged corpus) list matches with zero pining
token and zero network -- the fix for receiver_model's head-to-head re-fit.
"""

import json

import pytest

from scripts._loader_pining import PINING_MANIFEST_ENV, _list_matches


def test_list_matches_reads_staged_manifest_offline(tmp_path, monkeypatch):
    (tmp_path / "statsbomb").mkdir()
    (tmp_path / "statsbomb" / "matches.json").write_text(
        json.dumps({"matches": [{"id": "3986784", "visibility": "private", "artifacts": {}}]})
    )
    monkeypatch.setenv(PINING_MANIFEST_ENV, str(tmp_path))
    # A deliberately bogus base_url: offline mode must NOT touch the network. A real call would raise.
    out = _list_matches("statsbomb", "unused-token", "http://127.0.0.1:0/bogus")
    assert out == [{"id": "3986784", "visibility": "private", "artifacts": {}}]


def test_list_matches_offline_missing_manifest_is_loud(tmp_path, monkeypatch):
    monkeypatch.setenv(PINING_MANIFEST_ENV, str(tmp_path))  # dir exists, provider file absent
    with pytest.raises(FileNotFoundError, match="offline pining manifest for 'statsbomb'"):
        _list_matches("statsbomb", "unused", "http://unused")
