from scripts.hf_jobs.arm_policy import compute_arms
from scripts.hf_jobs.arms import match_key
from scripts.hf_jobs.datasets import pergame_name

# flat shard stems "{provider}__{game}" (the staged dataset's filenames). The Bundesliga shards are staged
# under the dataset IDENTITY name "idsse" (see silly-kicks/hf-refit-xshot17 `_actions/idsse__DFL-MAT-*`),
# but the in-job frames carry the SPADL converter source_provider "sportec", so the in-job match key is
# "sportec__idsse-DFL-MAT-J03WMX". The emitted arm key must equal THAT (else the DFL rows drop from the arm).
_BUNDESLIGA = "idsse__DFL-MAT-J03WMX"
_BUNDESLIGA_INJOB = "sportec__idsse-DFL-MAT-J03WMX"
_SC_PUBLIC = "skillcorner__1886347"
_SC_OWNER = "skillcorner__9999"


def test_public_arm_uses_injob_dataset_keys_not_registry_identity():
    payload = compute_arms([_BUNDESLIGA, _SC_PUBLIC, _SC_OWNER])
    public = set(payload["arms"]["public"])
    # the key's FIRST component is the in-job source_provider (converter "sportec"), NEVER the identity "idsse"
    assert _BUNDESLIGA_INJOB in public
    assert all(k.split("__")[0] != "idsse" for k in public)
    assert "skillcorner__skillcorner-1886347" in public
    assert "skillcorner__skillcorner-9999" not in public  # unregistered -> owner/private


def test_all_public_corpus_single_public_arm():
    payload = compute_arms([_BUNDESLIGA, _SC_PUBLIC])  # both registered public
    assert payload["ship_order"] == ["public"]
    assert set(payload["arms"]["public"]) == {_BUNDESLIGA_INJOB, "skillcorner__skillcorner-1886347"}


def test_owner_skillcorner_forms_sc_extended_not_public():
    payload = compute_arms([_SC_PUBLIC, _SC_OWNER])
    assert set(payload["arms"]["public"]) == {"skillcorner__skillcorner-1886347"}
    assert set(payload["arms"]["sc_extended"]) == {"skillcorner__skillcorner-1886347", "skillcorner__skillcorner-9999"}
    assert "full" not in payload["arms"]  # no gradientsports -> full == sc_extended, omitted


def test_emitted_keys_match_the_injob_match_key():
    # the in-job trainer computes match_key(source_provider, repackage dir pergame_name(stem_provider, game))
    # skillcorner: the converter name EQUALS the identity, so the stem provider is the source_provider.
    payload = compute_arms([_SC_PUBLIC])
    assert payload["arms"]["public"] == [match_key("skillcorner", pergame_name("skillcorner", "1886347"))]
    # Bundesliga: the stem carries the identity "idsse"; the in-job source_provider is the converter "sportec"
    # (ground truth from the dataset lock), and the repackage dir is still identity-named "idsse-...". The
    # emitted key must equal the in-job match_key of those two.
    payload = compute_arms([_BUNDESLIGA])
    assert payload["arms"]["public"] == [match_key("sportec", pergame_name("idsse", "DFL-MAT-J03WMX"))]


def test_dfl_intended_membership_equals_the_injob_actual():
    # r6 B24: the whole chain for the Bundesliga rows. The launcher's arms-json gives adopt its INTENDED count
    # (provider = the key prefix); the trainer masks rows with arm_masks over the in-job match_key and counts
    # the ACTUAL by the raw source_provider. Both must agree for DFL, else adopt refuses a correct run (or the
    # public arm silently drops the DFL matches, the d1623f1 defect).
    import json

    import numpy as np

    from scripts._study_shared import contributing_units_by_provider
    from scripts.hf_jobs.adopt import _intended_units
    from scripts.hf_jobs.arms import arm_masks, parse_arms

    arms_json = json.dumps(compute_arms([_BUNDESLIGA, _SC_PUBLIC]), sort_keys=True)  # as campaign.py passes it
    providers = np.array(["sportec", "sportec", "skillcorner"])  # per training ROW (two DFL rows, one match)
    keys = [
        match_key("sportec", pergame_name("idsse", "DFL-MAT-J03WMX")),
        match_key("sportec", pergame_name("idsse", "DFL-MAT-J03WMX")),
        match_key("skillcorner", pergame_name("skillcorner", "1886347")),
    ]
    shipped = arm_masks(keys, parse_arms(arms_json))["public"]
    assert shipped.all()  # no DFL row drops from the public arm
    actual = contributing_units_by_provider(providers, np.array(keys), shipped)
    assert _intended_units(arms_json, "public") == actual == {"sportec": 1, "skillcorner": 1}
