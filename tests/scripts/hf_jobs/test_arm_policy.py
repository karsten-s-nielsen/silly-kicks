import types

import pytest

from scripts.hf_jobs.arm_policy import compute_arms


def _unit(stem: str, source_provider: str) -> dict:
    """A staged unit as the lock records it (`datasets.staged_units`)."""
    return {"stem": stem, "game": stem.partition("__")[2], "source_provider": source_provider}


# The Bundesliga match is staged under its licensing identity (stem `idsse__`) and was SPADL-converted by the
# sportec converter (`source_provider`), as in every locked dataset (HFU spec 16.1, F29).
_BUNDESLIGA = _unit("idsse__DFL-MAT-J03WMX", "sportec")
_SC_PUBLIC = _unit("skillcorner__1886347", "skillcorner")
_SC_OWNER = _unit("skillcorner__9999", "skillcorner")


def test_public_arm_uses_the_injob_keys():
    public = set(compute_arms([_BUNDESLIGA, _SC_PUBLIC, _SC_OWNER])["arms"]["public"])
    # the in-job key is match_key(source_provider, pergame_name(provider, game))
    assert "sportec__idsse-DFL-MAT-J03WMX" in public
    assert "skillcorner__skillcorner-1886347" in public
    assert "skillcorner__skillcorner-9999" not in public  # unregistered -> owner/private


def test_all_public_corpus_single_public_arm():
    payload = compute_arms([_BUNDESLIGA, _SC_PUBLIC])  # both registered public
    assert payload["ship_order"] == ["public"]
    assert set(payload["arms"]["public"]) == {"sportec__idsse-DFL-MAT-J03WMX", "skillcorner__skillcorner-1886347"}


def test_owner_skillcorner_forms_sc_extended_not_public():
    payload = compute_arms([_SC_PUBLIC, _SC_OWNER])
    assert set(payload["arms"]["public"]) == {"skillcorner__skillcorner-1886347"}
    assert set(payload["arms"]["sc_extended"]) == {"skillcorner__skillcorner-1886347", "skillcorner__skillcorner-9999"}
    assert "full" not in payload["arms"]  # no gradientsports -> full == sc_extended, omitted


@pytest.mark.parametrize("trainer", ["scripts.train_xshot_occurrence", "scripts.train_xcross_attempt"])
@pytest.mark.parametrize("dataset", ["xshot17", "xshot115", "tc3"])
def test_emitted_keys_equal_the_trainer_keys_on_every_locked_unit(trainer, dataset):
    """F29: every arm key is a key the trainer builds from the same locked unit, and every unit is in an arm."""
    from importlib import import_module

    from scripts.hf_jobs.arms import match_key
    from scripts.hf_jobs.datasets import staged_units

    ctx = types.SimpleNamespace(data_root=None, params={"dataset": dataset})
    trainer_keys = {match_key(r.prov, r.mid) for r in import_module(trainer)._extract_refs(ctx)}
    arms = compute_arms(staged_units(dataset))["arms"]
    assert set().union(*arms.values()) == trainer_keys


def test_the_public_corpus_arm_holds_all_17_matches():
    from scripts.hf_jobs.datasets import staged_units

    assert len(compute_arms(staged_units("xshot17"))["arms"]["public"]) == 17
