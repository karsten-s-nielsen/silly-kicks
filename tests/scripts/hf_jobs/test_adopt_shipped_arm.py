"""Corpus widening spec 7.1 (ADR-132): `tier` is the CHANNEL (fixed before the run: store key, Instance.id, manifest
matching); the new manifest field `shipped_arm` is what the run's rule chose. A committed entry without it reads as
its tier (today's meaning). Through the real campaign instances on a fixture `xshot187`, both rule outcomes."""

from __future__ import annotations

import dataclasses
import json
import pathlib

import pytest

from scripts import _corpus
from scripts.hf_jobs import adopt as A
from scripts.hf_jobs import campaign as C
from scripts.hf_jobs import datasets as D
from scripts.hf_jobs import manifest as mf
from scripts.hf_jobs.registry import TrainerSpec, spec_for

SHA = "abcdef1" + "0" * 33
_UNITS = [
    {"stem": "skillcorner__1874553", "game": "1874553", "source_provider": "skillcorner"},
    {"stem": "idsse__DFL-MAT-J03WMX", "game": "DFL-MAT-J03WMX", "source_provider": "sportec"},
    {"stem": "gradientsports__3812", "game": "3812", "source_provider": "gradientsports"},
    {"stem": "skillcorner__1021404", "game": "1021404", "source_provider": "skillcorner"},  # licensed
]
_PUBLIC_IDS = sorted([["gradientsports", "3812"], ["idsse", "DFL-MAT-J03WMX"], ["skillcorner", "1874553"]])


@pytest.fixture
def xshot187(tmp_path, monkeypatch):
    """A fixture `xshot187` (declared + locked) beside the committed datasets."""
    toml = D.DECLARED.read_text(encoding="utf-8") + (
        '\n[datasets.xshot187]\nrepo = "silly-kicks/hf-refit-xshot187"\nkind = "flat_tc3"\nraw = "restricted"\n'
        'provenance = "extended"\naccess = "mount"\nread_concurrency = 4\ndescription = "fixture"\n'
    )
    p = tmp_path / "datasets.toml"
    p.write_text(toml, encoding="utf-8")
    monkeypatch.setattr(D, "DECLARED", p)
    lock = json.loads(D.LOCK.read_text(encoding="utf-8"))
    lock["xshot187"] = {
        "repo": "silly-kicks/hf-refit-xshot187",
        "revision": "f" * 40,
        "kind": "flat_tc3",
        "generation": "g",
        "home_teams": True,
        "units": _UNITS,
        "n_units": len(_UNITS),
        "identity": D._identities(_UNITS),
        "public_ids": _PUBLIC_IDS,
    }
    real = D._read_lock
    monkeypatch.setattr(D, "_read_lock", lambda path=D.LOCK: lock if path == D.LOCK else real(path))
    camp = tmp_path / "refit.toml"
    camp.write_text(
        '[campaign]\nname = "widen"\nmode = "refit"\nbucket = "silly-kicks/hf-refit-scratch"\nconcurrency_cap = 4\n'
        'attempts = 3\n\n[[jobs]]\njob = "xshot_occurrence"\ntier = "public"\ndataset = "xshot187"\n'
        'variants = ["default"]\narms = ["public", "sc_extended"]\n\n[[jobs]]\njob = "xshot_occurrence"\n'
        'tier = "sc_extended"\ndataset = "xshot187"\nvariants = ["default"]\narms = ["sc_extended"]\n',
        encoding="utf-8",
    )
    return C.load_campaign(camp)


def _manifest(*, wheel_arm: str | None) -> mf.Manifest:
    wheel = mf.Artifact(
        id="w",
        destination="wheel:silly_kicks/tracking/_xshot_weights/default",
        job="xshot_occurrence",
        variant="default",
        tier="public",
        certification="owner" if wheel_arm == "sc_extended" else "public",
        dataset="xshot187",
        naming="pergame",
        corpus="digest" if wheel_arm == "sc_extended" else "public_ids",
        shipped_arm=wheel_arm,
    )
    hub = mf.Artifact(
        id="h",
        destination="hub:silly-kicks/xshot-occurrence-v1",
        job="xshot_occurrence",
        variant="default",
        tier="sc_extended",
        certification="owner",
        dataset="xshot187",
        naming="pergame",
        corpus="digest",
    )
    return mf.Manifest(events={}, artifacts={"w": wheel, "h": hub}, roots={}, base=mf.REPO_ROOT)


def test_entry_without_shipped_arm_reads_as_tier():
    man = mf.load()
    assert man.artifacts
    for a in man.artifacts.values():
        assert a.shipped_arm is None
        assert mf.effective_shipped_arm(a) == a.tier  # every committed entry behaves exactly as today
    a = man.artifacts["xshot_default"]
    assert mf.effective_shipped_arm(dataclasses.replace(a, shipped_arm="sc_extended")) == "sc_extended"


def test_wheel_and_hub_instances_never_cross(xshot187):
    insts = C.instances(xshot187, SHA)
    wheel = next(i for i in insts if i.tier == "public")
    hub = next(i for i in insts if i.tier == "sc_extended")
    assert wheel.run_key != hub.run_key and wheel.id != hub.id
    assert json.loads(wheel.args[wheel.args.index("--arms-json") + 1])["ship_order"] == ["public", "sc_extended"]
    assert json.loads(hub.args[hub.args.index("--arms-json") + 1])["ship_order"] == ["sc_extended"]
    for arm in ("public", "sc_extended"):
        man = _manifest(wheel_arm=arm)
        assert [a.id for a in A.artifacts_for(wheel, man)] == ["w"]
        assert [a.id for a in A.artifacts_for(hub, man)] == ["h"]


def _fresh(tmp_path, shipped: str) -> pathlib.Path:
    d = tmp_path / f"fresh-{shipped}"
    d.mkdir(exist_ok=True)
    (d / "metrics.json").write_text(
        json.dumps({"shipped_variant": shipped, "candidates": {shipped: {"metrics": {"pr_auc": 0.2}}}}),
        encoding="utf-8",
    )
    return d


def _ship_violations(a: mf.Artifact, fresh: pathlib.Path) -> list[str]:
    return [v for v in A.expectation_violations(a, fresh, {}, SHA) if "ships" in v]


def _never_run(*a, **k):
    raise AssertionError("the owner branch runs no subprocess")


def test_adopt_each_outcome(xshot187, tmp_path):
    spec = spec_for("xshot_occurrence")
    assert isinstance(spec, TrainerSpec)
    wheel = next(i for i in C.instances(xshot187, SHA) if i.tier == "public")

    # outcome `public`: public_ids, rehydrate with the LOCKED list, certification basis public
    w_pub = _manifest(wheel_arm="public").artifacts["w"]
    assert spec.certification_for(mf.effective_shipped_arm(w_pub)) == "public"
    assert A._public_staged(wheel, mf.effective_shipped_arm(w_pub)) is True
    assert not _ship_violations(w_pub, _fresh(tmp_path, "public"))
    digest = {
        "corpus_match_ids_sha256": "x",
        "corpus_n_matches": 3,
        "providers": ["gradientsports", "idsse", "skillcorner"],
    }
    out = _corpus.rehydrate_public_corpus_ids(digest, pairs=D.locked("xshot187")["public_ids"])
    assert out["corpus_match_ids"] == _PUBLIC_IDS
    with pytest.raises(ValueError, match="pinned public arm"):
        _corpus.rehydrate_public_corpus_ids(
            {**digest, "corpus_n_matches": 17}, pairs=D.locked("xshot187")["public_ids"]
        )

    # outcome `sc_extended`: digest, no rehydrate, the owner branch with basis cv_parity
    w_sc = _manifest(wheel_arm="sc_extended").artifacts["w"]
    assert w_sc.corpus == "digest"  # only corpus = public_ids rehydrates (adopt._bundle)
    assert spec.certification_for(mf.effective_shipped_arm(w_sc)) == "owner"
    assert A._public_staged(wheel, mf.effective_shipped_arm(w_sc)) is False
    fresh = _fresh(tmp_path, "sc_extended")
    assert not _ship_violations(w_sc, fresh)
    res = A._certify(spec, wheel, w_sc, fresh, json.loads((fresh / "metrics.json").read_text()), {}, run=_never_run)
    assert res["basis"] == "cv_parity"


def test_shipped_variant_must_match_shipped_arm(xshot187, tmp_path):
    w_pub = _manifest(wheel_arm="public").artifacts["w"]
    assert _ship_violations(w_pub, _fresh(tmp_path, "sc_extended"))
    # an sc_extended run on an entry that has no shipped_arm yet (before the adoption PR writes it) is refused
    w_none = _manifest(wheel_arm=None).artifacts["w"]
    assert _ship_violations(w_none, _fresh(tmp_path, "sc_extended"))
    assert not _ship_violations(w_none, _fresh(tmp_path, "public"))


def test_the_adoption_report_prints_the_shipped_arm():
    man = _manifest(wheel_arm="sc_extended")
    text = A._changelog_draft(man, ["w"], "widen", SHA)
    assert "tier `public`" in text and "shipped arm `sc_extended`" in text
