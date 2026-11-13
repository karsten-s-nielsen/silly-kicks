import ast

from scripts.hf_jobs.arms import ArmSpec
from scripts.train_xcross_attempt import _build_candidate_masks


def test_candidate_masks_from_arms_not_providers():
    keys = ["sportec__DFL-MAT-J03WMX", "skillcorner__1874553", "skillcorner__9999"]
    spec = ArmSpec(
        arms={"public": frozenset(keys[:2]), "sc_extended": frozenset(keys)},
        ship_order=("public", "sc_extended"),
    )
    masks = _build_candidate_masks(keys, spec)
    assert masks["public"].tolist() == [True, True, False]
    assert masks["sc_extended"].tolist() == [True, True, True]


def test_xcross_source_has_no_visibility_imports():
    src = open("scripts/train_xcross_attempt.py", encoding="utf-8").read()
    names = {a.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom) for a in n.names}
    banned = {"match_visibility", "is_public_row", "is_obtainable_row", "assert_public_corpus", "artifact_label"}
    assert not (names & banned)
    for registry in ("RAW_REDISTRIBUTABLE", "PUBLICLY_OBTAINABLE", "HELD_OUT_PROBE"):
        assert registry not in src, registry
    assert "match_visibility" not in src
