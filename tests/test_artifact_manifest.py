"""INV-9: artifacts are checked against EXPECTATIONS declared before a run (manifest + dataset lock),
never against values derived from the artifact itself (ADR-128 spec 5.7)."""

from pathlib import Path

import pytest

from scripts.hf_jobs import manifest as M
from scripts.hf_jobs.datasets import declared

MAN = M.load()


def test_wheel_population_equals_the_discovered_dirs():
    found = {
        p.as_posix()
        for p in Path("silly_kicks").glob("**/_*_weights/*")
        if p.is_dir() and not p.name.startswith((".", "_")) and p.name != "full"
    }
    found |= {p.as_posix() for p in Path("silly_kicks").glob("*/weights") if p.is_dir()}
    assert found == {a.path for a in MAN.artifacts.values() if a.kind == "wheel"}


def test_hub_population_equals_the_card_set():
    cards = {
        p.name.removesuffix("-model-card.md") for p in Path("docs/huggingface/model-cards").glob("*-model-card.md")
    }
    hub_cards = {
        Path(a.card).name.removesuffix("-model-card.md")
        for a in MAN.artifacts.values()
        if a.kind == "hub" and a.card and a.card.startswith("docs/")
    }
    assert cards == hub_cards


@pytest.mark.parametrize("aid", sorted(a.id for a in MAN.artifacts.values() if a.kind == "wheel" and a.commit_field))
def test_wheel_artifact_matches_its_expectations(aid):
    assert not M.violations(MAN.artifacts[aid])


def test_every_event_is_used_and_every_reference_resolves():
    used = {a.refit_event for a in MAN.artifacts.values() if a.refit_event}
    assert used == set(MAN.events)
    for a in MAN.artifacts.values():
        assert a.dataset is None or a.dataset in declared(), a.id
        assert a.wheel_sibling is None or a.wheel_sibling in MAN.artifacts, a.id
        assert a.mirror_of is None or a.mirror_of in MAN.artifacts, a.id


def test_the_checker_catches_a_wrong_corpus():
    """Anti-rot: an artifact whose corpus ids differ from the staging record is reported."""
    a = MAN.artifacts["xshot_default"]
    fake = {
        "metrics.json": {
            "shipped_variant": "public",
            "reproducibility": "public",
            "corpus_match_ids": [["idsse", "X"]],
        },
        "metadata.json": {
            "shipped_variant": "public",
            "provider_list": ["idsse", "skillcorner"],
            "training_commit": a.training_commit,
        },
    }
    assert any("corpus" in v for v in M.violations(a, read=fake.__getitem__))


def test_the_checker_catches_a_wrong_commit_and_a_wrong_expectation():
    """Anti-rot for the other two checks: a foreign training commit and a wrong declared value."""
    a = MAN.artifacts["ghost_default"]
    fake = {"metadata.json": {"training_commit": "0" * 40, "corpus_provenance": {"providers": ["x"], "n_games": 179}}}
    bad = M.violations(a, read=fake.__getitem__)
    assert any("training_commit" in v for v in bad) and any("providers" in v for v in bad)


def test_observed_fields_never_feed_an_expectation():
    """INV-9: no expectation key may name an observed field."""
    observed_keys = {"sha256", "metrics_summary", "env", "campaign", "adopted_at"}
    for a in MAN.artifacts.values():
        assert not ({k.partition(":")[2].split(".")[0] for k in a.expect} & observed_keys), a.id


def test_training_commit_exactly_when_commit_field():
    """HFU spec 18.2: `adopt` writes a manifest `training_commit` only for an artifact with a `commit_field`, so an
    entry declares a training commit exactly when it declares the bundle field that records it."""
    for a in MAN.artifacts.values():
        assert (a.training_commit is None) == (a.commit_field is None), a.id


def test_hub_class_names_the_loader_and_its_round_trip():
    """HFU fit-units spec C2: `post_verify` round-trips a Hub artifact with ``from_hub`` when it declares a
    ``hub_class``, and with the job loader's ``load`` when it declares none. So a declared class must BE the
    job loader and have ``from_hub``; an artifact without one must have a loader with ``load``."""
    import importlib

    from scripts.hf_jobs.adopt import LOADERS

    with_class, without = [], []
    for a in sorted((x for x in MAN.artifacts.values() if x.kind == "hub"), key=lambda x: x.id):
        mod, _, cls = LOADERS[a.job].partition(":")
        klass = getattr(importlib.import_module(mod), cls)
        if a.hub_class:
            assert cls == a.hub_class, f"{a.id}: hub_class {a.hub_class!r} is not the job loader {cls!r}"
            assert hasattr(klass, "from_hub"), f"{a.id}: {cls} has no from_hub"
            with_class.append(a.id)
        else:
            assert hasattr(klass, "load"), f"{a.id}: {cls} has no load"
            assert not hasattr(klass, "from_hub"), f"{a.id}: {cls} has from_hub; declare hub_class"
            without.append(a.id)
    assert with_class and without == ["hub_xsuccess"]
