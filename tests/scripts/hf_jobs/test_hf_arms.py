import json

import pytest

from scripts.hf_jobs.arms import ArmSpec, ArmsUnavailableError, arm_masks, load_arms, match_key


def test_match_key_is_provider_double_underscore_id():
    assert match_key("sportec", "DFL-MAT-J03WMX") == "sportec__DFL-MAT-J03WMX"


def test_load_arms_round_trips(tmp_path):
    p = tmp_path / "arms.json"
    p.write_text(
        json.dumps(
            {
                "arms": {
                    "public": ["sportec__DFL-MAT-J03WMX", "skillcorner__1874553"],
                    "sc_extended": ["sportec__DFL-MAT-J03WMX", "skillcorner__1874553", "skillcorner__9999"],
                },
                "ship_order": ["public", "sc_extended"],
            }
        )
    )
    spec = load_arms(p)
    assert spec.ship_order == ("public", "sc_extended")
    assert spec.arms["public"] == frozenset({"sportec__DFL-MAT-J03WMX", "skillcorner__1874553"})


def test_arm_masks_membership_by_dataset_key():
    keys = ["sportec__DFL-MAT-J03WMX", "skillcorner__1874553", "skillcorner__9999"]
    spec = ArmSpec(arms={"public": frozenset(keys[:2]), "full": frozenset(keys)}, ship_order=("public", "full"))
    m = arm_masks(keys, spec)
    assert m["public"].tolist() == [True, True, False]
    assert m["full"].tolist() == [True, True, True]


def test_load_arms_missing_file_raises(tmp_path):
    with pytest.raises(ArmsUnavailableError):
        load_arms(tmp_path / "nope.json")


def test_load_arms_all_empty_raises(tmp_path):
    p = tmp_path / "arms.json"
    p.write_text(json.dumps({"arms": {"public": []}, "ship_order": ["public"]}))
    with pytest.raises(ArmsUnavailableError):
        load_arms(p)


def test_an_arm_key_naming_no_unit_of_the_dataset_is_refused():
    """Fail closed (HFU R4-10, 2026-10-30): an arm key that names no unit of the dataset the job reads is a key
    mismatch between the launcher and the job, and the row filter would drop its rows silently. The refusal names
    the stray keys; a spec whose keys all name units passes."""
    from scripts.hf_jobs.arms import ArmKeyMismatchError, require_arm_keys_in

    spec = ArmSpec(
        arms={"public": frozenset({"sportec__idsse-g1"}), "sc_extended": frozenset({"sportec__idsse-g1", "x__x-2"})},
        ship_order=("public", "sc_extended"),
    )
    require_arm_keys_in(spec, ["sportec__idsse-g1", "x__x-2", "x__x-3"])  # a unit outside every arm is fine
    stray = ArmSpec(arms={"public": frozenset({"idsse__idsse-g1", "x__x-2"})}, ship_order=("public",))
    with pytest.raises(ArmKeyMismatchError, match="idsse__idsse-g1"):
        require_arm_keys_in(stray, ["sportec__idsse-g1", "x__x-2"])
