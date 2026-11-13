"""Spec 8.3 / ADR-038 / ADR-132: every TF-58 artifact carries the corpus-visibility (provenance) label of the matches
its aggregates came from.

The label is ``public`` iff every used match is publicly obtainable (the ``PUBLICLY_OBTAINABLE`` registry, never the
provider name), fail-closed (an unregistered match is not obtainable). The pining manifest's per-match ``visibility``
stays the licensing check: a match may claim ``public`` there only if it is raw-redistributable.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts import _coordination_corpus as cc
from scripts._corpus import RAW_REDISTRIBUTABLE

_PUBLIC_IDSSE = [("idsse", m) for m in sorted(RAW_REDISTRIBUTABLE["idsse"])[:2]]


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


def test_a_gs_match_is_obtainable_so_the_artifact_stays_public(monkeypatch):
    # ADR-132: GS is pining-private (raw not redistributable) but publicly obtainable -> `public` provenance.
    _manifest(monkeypatch, {**{k: "public" for k in _PUBLIC_IDSSE}, ("gradientsports", "3812"): "private"})
    assert cc.corpus_visibility_label([*_PUBLIC_IDSSE, ("gradientsports", "3812")], token=None) == "public"


def test_one_licensed_match_makes_the_artifact_non_public(monkeypatch):
    _manifest(monkeypatch, {**{k: "public" for k in _PUBLIC_IDSSE}, ("skillcorner", "1021404"): "private"})
    assert cc.corpus_visibility_label([*_PUBLIC_IDSSE, ("skillcorner", "1021404")], token=None) == "sc_extended"


def test_an_unregistered_match_is_not_obtainable(monkeypatch):
    # fail-closed: provenance reads the registry, not the manifest; an unregistered match is never public.
    _manifest(monkeypatch, {})
    assert cc.corpus_visibility_label([*_PUBLIC_IDSSE, ("gradientsports", "99999")], token=None) == "sc_extended"
    assert cc.corpus_visibility_label(_PUBLIC_IDSSE, token=None) == "public"  # registered, manifest silent


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
