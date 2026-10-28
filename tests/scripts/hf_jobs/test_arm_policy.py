from scripts.hf_jobs.arm_policy import compute_arms


def _unit(stem: str, source_provider: str | None = None) -> dict:
    """A locked unit (`datasets.staged_units`): its flat shard stem ``{provider}__{game}`` and its recorded
    ``source_provider`` (by default the stem's provider)."""
    return {"stem": stem, "source_provider": source_provider or stem.partition("__")[0]}


# The Bundesliga match staged under the sportec converter name (the registry keys it under idsse); the real locks
# stage it as `idsse__<game>` with source_provider `sportec` (`_DFL_STAGED`, HFU R4-10).
_BUNDESLIGA = _unit("sportec__DFL-MAT-J03WMX")
_DFL_STAGED = _unit("idsse__DFL-MAT-J03WMX", "sportec")
_SC_PUBLIC = _unit("skillcorner__1886347")
_SC_OWNER = _unit("skillcorner__9999")


def test_public_arm_uses_injob_dataset_keys_not_registry_identity():
    payload = compute_arms([_BUNDESLIGA, _SC_PUBLIC, _SC_OWNER])
    public = set(payload["arms"]["public"])
    # emitted as the IN-JOB match key {provider}__{provider}-{game}, NOT the registry identity idsse__
    assert "sportec__sportec-DFL-MAT-J03WMX" in public
    assert not any("idsse" in k for k in public)
    assert "skillcorner__skillcorner-1886347" in public
    assert "skillcorner__skillcorner-9999" not in public  # unregistered -> owner/private


def test_all_public_corpus_single_public_arm():
    payload = compute_arms([_BUNDESLIGA, _SC_PUBLIC])  # both registered public
    assert payload["ship_order"] == ["public"]
    assert set(payload["arms"]["public"]) == {"sportec__sportec-DFL-MAT-J03WMX", "skillcorner__skillcorner-1886347"}


def test_owner_skillcorner_forms_sc_extended_not_public():
    payload = compute_arms([_SC_PUBLIC, _SC_OWNER])
    assert set(payload["arms"]["public"]) == {"skillcorner__skillcorner-1886347"}
    assert set(payload["arms"]["sc_extended"]) == {"skillcorner__skillcorner-1886347", "skillcorner__skillcorner-9999"}
    assert "full" not in payload["arms"]  # no gradientsports -> full == sc_extended, omitted


def test_emitted_keys_match_the_injob_match_key():
    # the in-job trainer computes match_key(source_provider, repackage dir "{provider}-{game}")
    from scripts.hf_jobs.arms import match_key

    payload = compute_arms([_SC_PUBLIC])
    assert payload["arms"]["public"] == [match_key("skillcorner", "skillcorner-1886347")]


def test_a_unit_staged_under_its_identity_is_keyed_by_its_source_provider():
    """HFU R4-10: the real locks stage the DFL matches as ``idsse__<game>`` with ``source_provider`` ``sportec``;
    the job keys their rows ``sportec__idsse-<game>``, so the launcher must emit exactly that key, and the match is
    public (the registry lookup goes by the stem's provider)."""
    payload = compute_arms([_DFL_STAGED, _SC_PUBLIC, _SC_OWNER])
    assert "sportec__idsse-DFL-MAT-J03WMX" in payload["arms"]["public"]
    assert "idsse__idsse-DFL-MAT-J03WMX" not in {k for keys in payload["arms"].values() for k in keys}
    assert "sportec__idsse-DFL-MAT-J03WMX" in payload["arms"]["sc_extended"]


def test_a_tier_receives_only_the_arms_it_ships_at_most():
    """Spec r4 D18 (c): tiers that read one dataset share it, so the launcher gives each tier only its arms --
    public -> public; sc_extended -> public + sc_extended -- in ship order. An unknown tier refuses, named."""
    import pytest

    from scripts.hf_jobs.arm_policy import arms_for_tier

    full = compute_arms([_SC_PUBLIC, _SC_OWNER, _unit("gradientsports__10502")])
    assert set(full["arms"]) == {"public", "sc_extended", "full"}  # non-vacuous: three arms to restrict
    assert arms_for_tier(full, "public") == {"arms": {"public": full["arms"]["public"]}, "ship_order": ["public"]}
    sc = arms_for_tier(full, "sc_extended")
    assert set(sc["arms"]) == {"public", "sc_extended"} and sc["ship_order"] == ["public", "sc_extended"]
    with pytest.raises(ValueError, match="owner"):
        arms_for_tier(full, "owner")


def test_every_arm_key_is_the_injob_key_of_a_locked_unit():
    """The launcher's ``--arms-json`` must name rows by the key the trainer computes in-job,
    ``match_key(source_provider, pergame_name(provider, game))``, over the LOCKED dataset's units; the public arm
    holds every unit the registry calls public. The staged DFL matches are ``idsse__<game>`` shards whose
    ``source_provider`` is ``sportec`` (HFU R4-10, 2026-10-30): keys built from the stem alone named no row, so
    every arm silently lost the 7 public DFL matches."""
    import json
    import pathlib

    from scripts._driver_sources import staged_flat_refs
    from scripts.hf_jobs import arm_policy
    from scripts.hf_jobs import campaign as C
    from scripts.hf_jobs.arms import match_key
    from scripts.hf_jobs.datasets import pergame_name
    from scripts.hf_jobs.registry import spec_for

    c = C.load_campaign(pathlib.Path("scripts/hf_jobs/campaigns/cycle1.toml"))
    checked = 0
    for inst in C.instances(c, "abcdef1" + "0" * 33):
        if not getattr(spec_for(inst.job), "visibility_gated", False):
            continue
        refs = staged_flat_refs(None, inst.args[inst.args.index("--dataset") + 1])
        injob = {match_key(r.source_provider, pergame_name(r.provider, r.game)): r for r in refs}
        arms = json.loads(inst.args[inst.args.index("--arms-json") + 1])["arms"]
        for name, keys in arms.items():
            assert set(keys) <= set(injob), (inst.job, inst.tier, name, sorted(set(keys) - set(injob))[:3])
        public = {k for k, r in injob.items() if arm_policy._is_public(r.provider, r.game)}
        assert set(arms["public"]) == public, (inst.job, inst.tier, sorted(public - set(arms["public"]))[:3])
        assert any(r.source_provider != r.provider for r in injob.values())  # non-vacuous: the DFL staging
        checked += 1
    assert checked == 8  # xshot + xcross, two tiers, two variants
