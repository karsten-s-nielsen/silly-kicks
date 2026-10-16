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
