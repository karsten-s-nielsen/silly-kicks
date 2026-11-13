"""Spec 3 / 4 (corpus widening, ADR-132): the three corpus registries and the provenance label.

Each registry is checked against ONE derived source (ADR-056), never a second hand-list: the GS part of
``PUBLICLY_OBTAINABLE`` against the GS units of the committed ``tc3`` lock, ``HELD_OUT_PROBE`` against the committed
``xcross_probe`` units. ``RAW_REDISTRIBUTABLE`` keeps today's live pining check and ``test_corpus_guard.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from scripts import _corpus
from scripts._corpus import (
    HELD_OUT_PROBE,
    PUBLICLY_OBTAINABLE,
    RAW_REDISTRIBUTABLE,
    artifact_label,
    assert_public_corpus,
    is_obtainable_row,
    unit_identity,
)
from scripts.hf_jobs import datasets as D

_REPO = Path(__file__).resolve().parents[2]


def test_gs_obtainable_equals_the_tc3_lock():
    games = {unit_identity(u) for u in D.staged_units("tc3")}
    gs = {g for p, g in games if p == "gradientsports"}
    assert PUBLICLY_OBTAINABLE["gradientsports"] == frozenset(gs)
    assert len(gs) == 64


def test_probe_equals_the_xcross_probe_lock():
    probe = {(p, g) for p, ids in HELD_OUT_PROBE.items() for g in ids}
    assert probe == {unit_identity(u) for u in D.staged_units("xcross_probe")}
    assert probe == {("gradientsports", "10502"), ("gradientsports", "10503")}


def test_unit_identity_normalises_provider_and_strips_the_pergame_prefix():
    assert unit_identity({"stem": "idsse__DFL-MAT-J03WMX", "game": "DFL-MAT-J03WMX", "source_provider": "sportec"}) == (
        "idsse",
        "DFL-MAT-J03WMX",
    )
    pergame = {"stem": "gradientsports-10502", "game": "gradientsports-10502", "source_provider": "gradientsports"}
    assert unit_identity(pergame) == ("gradientsports", "10502")


def test_raw_is_inside_obtainable():
    for prov, ids in RAW_REDISTRIBUTABLE.items():
        assert ids <= PUBLICLY_OBTAINABLE[prov]
    assert set(PUBLICLY_OBTAINABLE) == {"skillcorner", "idsse", "gradientsports"}
    assert PUBLICLY_OBTAINABLE["skillcorner"] == RAW_REDISTRIBUTABLE["skillcorner"]
    assert PUBLICLY_OBTAINABLE["idsse"] == RAW_REDISTRIBUTABLE["idsse"]


def test_probe_is_obtainable():
    for prov, ids in HELD_OUT_PROBE.items():
        assert ids <= PUBLICLY_OBTAINABLE[prov]


def test_unregistered_is_neither():
    providers = np.array(["skillcorner", "gradientsports", "sportec", "skillcorner", "gradientsports", "metrica"])
    match_ids = np.array(["9999", "3812", "DFL-MAT-J03WMX", "1874553", "99999", "1"])
    got = is_obtainable_row(providers=providers, match_ids=match_ids)
    assert got.tolist() == [False, True, True, True, False, False]
    assert "9999" not in RAW_REDISTRIBUTABLE["skillcorner"]


def test_unregistered_public_still_refuses():
    # A GS match is obtainable, but it is NOT raw-redistributable: a pining record that calls it `public` is the
    # licensing failure assert_public_corpus exists to stop (spec test 2).
    with pytest.raises(SystemExit, match="UNREGISTERED"):
        assert_public_corpus({("gradientsports", "3812"): "public"})


def test_labels():
    assert artifact_label(all_obtainable=True) == "public"  # GS-only, or GS + public SkillCorner
    assert artifact_label(all_obtainable=False) == "sc_extended"  # any licensed SkillCorner row
    gs_only = is_obtainable_row(providers=np.array(["gradientsports"] * 2), match_ids=np.array(["3812", "10517"]))
    gs_sc = is_obtainable_row(
        providers=np.array(["gradientsports", "skillcorner"]), match_ids=np.array(["3812", "1874553"])
    )
    licensed = is_obtainable_row(
        providers=np.array(["gradientsports", "skillcorner"]), match_ids=np.array(["3812", "1021404"])
    )
    assert artifact_label(all_obtainable=bool(gs_only.all())) == "public"
    assert artifact_label(all_obtainable=bool(gs_sc.all())) == "public"
    assert artifact_label(all_obtainable=bool(licensed.all())) == "sc_extended"
    assert {artifact_label(all_obtainable=v) for v in (True, False)} == {"public", "sc_extended"}  # never `full`


_COMMITTED_FULL = (
    "docs/research/tf58_team_coordination/metrics.json",
    "docs/research/tf58_team_coordination/numerics_noflip.json",
    "docs/research/tf58_team_coordination/derivation.json",
    "docs/research/tf58_team_coordination/calibration.json",
    "docs/research/cover_shadow_rq1/metrics.json",
    "docs/research/pass_risk_calibration/metrics.json",
)


@pytest.mark.parametrize("rel", _COMMITTED_FULL)
def test_committed_full_records_untouched(rel):
    # Q-D: no committed record is regenerated; the retired label stays readable where it was written.
    assert json.loads((_REPO / rel).read_text(encoding="utf-8"))["corpus_visibility"] == "full"


def test_full_label_still_renders(monkeypatch):
    # Q-A "loads and verifies": no code validates the label vocabulary; the one reader is the card renderer.
    from scripts.hf_jobs import manifest as M

    a = SimpleNamespace(id="x")
    monkeypatch.setattr(M, "_card_artifact", lambda card, man: a)
    monkeypatch.setattr(M, "_metrics", lambda art, base=None: {"run_commit": "a" * 40, "corpus_visibility": "full"})
    monkeypatch.setattr(M, "_event", lambda man, art: SimpleNamespace(version="4.0.0"))
    out = M.render_region("provenance", "card.md", cast(Any, SimpleNamespace(base=_REPO)))
    assert "`corpus_visibility: full`" in out


def test_module_keeps_no_old_name():
    assert not hasattr(_corpus, "PUBLIC_CORPUS")
