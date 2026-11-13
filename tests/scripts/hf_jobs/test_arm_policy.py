import types

import pytest

from scripts._corpus import HELD_OUT_PROBE, unit_identity
from scripts.hf_jobs.arm_policy import _injob_key, compute_arms, restrict_arms
from scripts.hf_jobs.datasets import staged_units


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
    """F29: every arm key is a key the trainer builds from the same locked unit; every non-probe unit is in an arm."""
    from importlib import import_module

    from scripts.hf_jobs.arms import match_key
    from scripts.hf_jobs.datasets import staged_units

    ctx = types.SimpleNamespace(data_root=None, params={"dataset": dataset})
    trainer_keys = {match_key(r.prov, r.mid) for r in import_module(trainer)._extract_refs(ctx)}
    units = staged_units(dataset)
    probe_keys = {_injob_key(u) for u in units if _is_probe(u)}
    arms = compute_arms(units)["arms"]
    in_arms = set().union(*arms.values())
    # spec 5.1 / 7.2: every NON-probe unit is in an arm, and the probe units (tc3 holds 10502, 10503) are in none
    assert in_arms == trainer_keys - probe_keys
    assert not (in_arms & probe_keys)
    assert (dataset == "tc3") == bool(probe_keys)


def test_the_public_corpus_arm_holds_all_17_matches():
    from scripts.hf_jobs.datasets import staged_units

    assert len(compute_arms(staged_units("xshot17"))["arms"]["public"]) == 17


def _is_probe(unit) -> bool:
    prov, game = unit_identity(unit)
    return game in HELD_OUT_PROBE.get(prov, frozenset())


_GS = _unit("gradientsports__3812", "gradientsports")
_GS_PROBE = _unit("gradientsports__10502", "gradientsports")
_SC_PUBLIC_NEW = _unit("skillcorner__1874553", "skillcorner")  # one of the 10 newer public SkillCorner matches


def test_public_is_obtainable_minus_probe():
    payload = compute_arms([_BUNDESLIGA, _SC_PUBLIC, _SC_PUBLIC_NEW, _GS, _GS_PROBE, _SC_OWNER])
    assert set(payload["arms"]["public"]) == {_injob_key(u) for u in (_BUNDESLIGA, _SC_PUBLIC, _SC_PUBLIC_NEW, _GS)}


def test_sc_extended_adds_licensed_skillcorner():
    payload = compute_arms([_BUNDESLIGA, _GS, _SC_OWNER])
    assert set(payload["arms"]["sc_extended"]) == {_injob_key(u) for u in (_BUNDESLIGA, _GS, _SC_OWNER)}
    assert payload["ship_order"] == ["public", "sc_extended"]


def test_no_full_arm():
    payload = compute_arms(staged_units("tc3"))  # 64 GS matches, 98 licensed SkillCorner
    assert set(payload["arms"]) == {"public", "sc_extended"}
    assert "full" not in payload["ship_order"]


def test_probe_unit_in_no_arm():
    payload = compute_arms([_GS_PROBE, _GS])
    assert _injob_key(_GS_PROBE) not in set().union(*payload["arms"].values())
    assert payload["arms"] == {"public": [_injob_key(_GS)]}


def test_an_unregistered_non_skillcorner_unit_is_refused():
    # fail loud: a unit neither obtainable nor licensed SkillCorner has no arm; dropping it silently would hide it.
    with pytest.raises(ValueError, match="no arm"):
        compute_arms([_GS, _unit("gradientsports__99999", "gradientsports")])


def test_arms_follow_staged_units():
    units = staged_units("tc3")
    full = compute_arms(units)["arms"]
    assert (len(full["public"]), len(full["sc_extended"])) == (79, 177)  # SC 10 + IDSSE 7 + GS 62; + 98 licensed
    dropped = [u for u in units if u["stem"] != "gradientsports__3812"]  # e.g. an S1 exclusion (spec 6.4)
    fewer = compute_arms(dropped)["arms"]
    assert (len(fewer["public"]), len(fewer["sc_extended"])) == (78, 176)


def test_restrict_both_arms_runs_paired():
    units = staged_units("tc3")
    payload = compute_arms(units)
    got = restrict_arms(payload, ["public", "sc_extended"], units)
    assert got == payload


def test_restrict_keeps_the_ship_order_of_the_job_arms():
    units = staged_units("xshot115")
    got = restrict_arms(compute_arms(units), ["sc_extended"], units)
    assert list(got["arms"]) == ["sc_extended"] and got["ship_order"] == ["sc_extended"]


def test_restrict_unknown_arm_refused():
    units = staged_units("xshot17")
    with pytest.raises(ValueError, match="sc_extended"):
        restrict_arms(compute_arms(units), ["sc_extended"], units)
    with pytest.raises(ValueError, match="non-empty"):
        restrict_arms(compute_arms(units), [], units)


def test_single_arm_strict_subset_is_refused():
    # spec 5.5: a one-arm job studies and gates on every dataset row but fits on its arm's rows -> refuse a subset.
    units = staged_units("xshot115")
    with pytest.raises(ValueError, match="strict subset"):
        restrict_arms(compute_arms(units), ["public"], units)


def test_single_arm_missing_only_the_probe_is_refused():
    # Review A CW-IMPL-02 (spec 5.5, test 11: a strict subset of the dataset's units): a one-arm job on tc3 misses
    # only the probe units, yet its `single` study and gate CV would read them while its fit leaves them out.
    units = staged_units("tc3")
    with pytest.raises(ValueError, match="strict subset"):
        restrict_arms(compute_arms(units), ["sc_extended"], units)


def test_committed_shapes_pass():
    u17, u115 = staged_units("xshot17"), staged_units("xshot115")
    assert list(compute_arms(u17)["arms"]) == ["public"]
    assert restrict_arms(compute_arms(u17), ["public"], u17) == compute_arms(u17)
    assert list(restrict_arms(compute_arms(u115), ["sc_extended"], u115)["arms"]) == ["sc_extended"]
