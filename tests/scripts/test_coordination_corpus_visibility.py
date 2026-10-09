"""Spec 8.3 / ADR-038: every TF-58 artifact carries the corpus-visibility label of the matches its aggregates came from.

The label is keyed on the pining manifest's per-match ``visibility`` (never the provider name), fail-closed (a match the
manifest does not list is private), and a match may claim ``public`` only if it is one of the registered public set.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts import _coordination_corpus as cc
from scripts._corpus import PUBLIC_CORPUS

_PUBLIC_IDSSE = [("idsse", m) for m in sorted(PUBLIC_CORPUS["idsse"])[:2]]


def _manifest(monkeypatch, visibility):
    seen = {}

    def fake(providers, *, token=None, base_url=None):
        seen["providers"] = sorted(providers)
        return dict(visibility)

    monkeypatch.setattr("scripts._loader_pining.match_visibility", fake)
    return seen


def test_public_only_when_every_used_match_is_a_registered_public_match(monkeypatch):
    seen = _manifest(monkeypatch, {k: "public" for k in _PUBLIC_IDSSE})
    assert cc.corpus_visibility_label(_PUBLIC_IDSSE, token=None) == "public"
    assert seen["providers"] == ["idsse"]  # the manifest is asked about the providers actually used


def test_one_owner_tier_match_makes_the_artifact_non_public(monkeypatch):
    _manifest(monkeypatch, {**{k: "public" for k in _PUBLIC_IDSSE}, ("gradientsports", "3812"): "private"})
    assert cc.corpus_visibility_label([*_PUBLIC_IDSSE, ("gradientsports", "3812")], token=None) == "full"


def test_a_match_the_manifest_does_not_list_is_private(monkeypatch):
    _manifest(monkeypatch, {})  # fail-closed: absent -> private, so never "public"
    assert cc.corpus_visibility_label(_PUBLIC_IDSSE, token=None) == "sc_extended"


def test_an_unregistered_public_claim_is_refused(monkeypatch):
    _manifest(monkeypatch, {("skillcorner", "999"): "public"})
    with pytest.raises(SystemExit, match="UNREGISTERED"):
        cc.corpus_visibility_label([("skillcorner", "999")], token=None)


def test_table_pairs_read_the_match_or_game_column():
    tables = [
        pd.DataFrame({"provider": ["idsse", "idsse"], "match_id": ["a", "a"]}),
        pd.DataFrame({"provider": ["skillcorner"], "game_id": ["g"]}),
        pd.DataFrame(),
    ]
    assert cc.table_pairs(*tables) == {("idsse", "a"), ("skillcorner", "g")}
