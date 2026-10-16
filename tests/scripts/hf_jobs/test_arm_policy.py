from scripts.hf_jobs.arm_policy import compute_arms

# flat shard stems "{provider}__{game}" (the staged dataset's filenames); the Bundesliga match is staged
# under the sportec converter name, the registry keys it under idsse.
_BUNDESLIGA = "sportec__DFL-MAT-J03WMX"
_SC_PUBLIC = "skillcorner__1886347"
_SC_OWNER = "skillcorner__9999"


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
