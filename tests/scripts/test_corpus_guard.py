"""G1 corpus guard + corpus identity + reproducibility helpers (combined-cycle-completion spec 5, 0.11)."""

import json
import pathlib

import pytest

from scripts._corpus import (
    BUNDLED_PUBLIC_ARM,
    PUBLIC_CORPUS,
    bundled_public_arm_pairs,
    rehydrate_public_corpus_ids,
    requested_is_all_public,
)
from scripts._corpus_identity import (
    check_expected_variant,
    check_shipped_variant,
    corpus_identity,
    match_id_pairs,
    reproducibility,
)


def test_bundled_public_arm_is_the_original_17_inside_the_registered_public_corpus():
    assert len(BUNDLED_PUBLIC_ARM["skillcorner"]) == 10
    assert len(BUNDLED_PUBLIC_ARM["idsse"]) == 7
    for prov, ids in BUNDLED_PUBLIC_ARM.items():
        assert set(ids) <= PUBLIC_CORPUS[prov]
    # 1874553 is one of the ten SkillCorner ids added on 2026-09-10 (ce0401a) -- not in the bundled arm
    assert "1874553" in PUBLIC_CORPUS["skillcorner"]
    assert "1874553" not in BUNDLED_PUBLIC_ARM["skillcorner"]
    assert set(BUNDLED_PUBLIC_ARM["idsse"]) == PUBLIC_CORPUS["idsse"]


def test_bundled_public_arm_pairs_shape():
    pairs = bundled_public_arm_pairs()
    assert len(pairs) == 17
    assert pairs == sorted(pairs)
    assert ["skillcorner", "1886347"] in pairs
    assert ["idsse", "DFL-MAT-J03WMX"] in pairs
    assert bundled_public_arm_pairs(("skillcorner",)) == [
        ["skillcorner", m] for m in sorted(BUNDLED_PUBLIC_ARM["skillcorner"])
    ]


def test_match_id_pairs_dedups_sorts_and_stringifies():
    assert match_id_pairs(["sk", "sk", "id"], [2, 2, "a"]) == [["id", "a"], ["sk", "2"]]


@pytest.mark.parametrize(
    ("pairs", "vis", "want"),
    [
        ([], {}, False),  # an empty request is never public
        ([("sk", "1")], {("sk", "1"): "public"}, True),
        ([("sk", "1"), ("sk", "2")], {("sk", "1"): "public", ("sk", "2"): "private"}, False),
        ([("sk", "1")], {}, False),  # absent from the manifest -> restricted (fail-closed)
    ],
)
def test_requested_is_all_public(pairs, vis, want):
    assert requested_is_all_public(pairs, vis) is want


def test_check_expected_variant_public_refuses_a_restricted_request():
    with pytest.raises(SystemExit, match="non-public"):
        check_expected_variant("public", all_public=False)


@pytest.mark.parametrize(
    ("expected", "all_public"), [(None, False), ("public", True), ("sc_extended", False), ("full", False)]
)
def test_check_expected_variant_passes(expected, all_public):
    check_expected_variant(expected, all_public=all_public)


def test_check_shipped_variant():
    check_shipped_variant(None, "sc_extended")
    check_shipped_variant("public", "public")
    with pytest.raises(SystemExit, match="would ship 'sc_extended'"):
        check_shipped_variant("public", "sc_extended")


def test_corpus_identity_records_ids_only_for_an_all_public_corpus():
    pub = corpus_identity(["sk", "sk"], ["1", "2"], all_public=True)
    assert pub == {"corpus_match_ids": [["sk", "1"], ["sk", "2"]]}
    restricted = corpus_identity(["sk", "sk"], ["1", "999"], all_public=False)
    assert set(restricted) == {"corpus_match_ids_sha256", "corpus_n_matches"}
    assert restricted["corpus_n_matches"] == 2
    assert "999" not in str(restricted)  # no id leaks into a restricted artifact
    # the digest is a function of the identity, so two runs on one corpus agree and different corpora differ
    assert corpus_identity(["sk"], ["999"], all_public=False) != restricted
    assert corpus_identity(["sk", "sk"], ["999", "1"], all_public=False) == restricted


def test_rehydrate_public_corpus_ids_replaces_the_digest_with_canonical_pairs():
    # A visibility-blind trainer artifact carries the DIGEST, not the ids (ADR-123 Amendment B). The
    # off-HF adoption seam rehydrates the canonical public ids, in place of the digest, position-preserved.
    blind = {
        "shipped_variant": "public",
        "providers": ["idsse", "skillcorner"],
        "corpus_match_ids_sha256": "7f8151127456de5a9152cd8bdbb92b9277159557769d180dcfbd7e74f5d076db",
        "corpus_n_matches": 17,
        "reproducibility": "public",
    }
    out = rehydrate_public_corpus_ids(blind)
    assert out["corpus_match_ids"] == bundled_public_arm_pairs()
    assert "corpus_match_ids_sha256" not in out and "corpus_n_matches" not in out
    assert list(out).index("corpus_match_ids") == list(blind).index("corpus_match_ids_sha256")  # in place
    assert blind.get("corpus_match_ids") is None  # input not mutated
    assert rehydrate_public_corpus_ids(out) == out  # idempotent on an already-canonical bundle


def test_rehydrate_public_corpus_ids_refuses_a_non_pinned_corpus():
    # A different count (e.g. the widened current PUBLIC_CORPUS, D4) or different providers is refused,
    # never silently relabelled as the pinned public arm.
    with pytest.raises(ValueError, match="pinned public arm"):
        rehydrate_public_corpus_ids({"corpus_n_matches": 25, "providers": ["idsse", "skillcorner"]})
    with pytest.raises(ValueError, match="pinned public providers"):
        rehydrate_public_corpus_ids({"corpus_n_matches": 17, "providers": ["idsse", "skillcorner", "statsbomb"]})


def test_committed_xshot_bundles_carry_the_canonical_public_corpus_ids():
    # The adopted wheel bundles must carry the explicit canonical public ids (the output of the seam) --
    # the corpus-policy gate's value, asserted here co-located with the seam that produces it.
    root = pathlib.Path("silly_kicks/tracking")
    for d in ("_xshot_weights/default", "_xshot_weights/position_only"):
        m = json.loads((root / d / "metrics.json").read_text(encoding="utf-8"))
        assert m["corpus_match_ids"] == bundled_public_arm_pairs(), d


def test_reproducibility_public_and_restricted():
    assert reproducibility("public", ["idsse", "skillcorner"]) == {"reproducibility": "public"}
    r = reproducibility("sc_extended", ["skillcorner"], training_commit="abc1234")
    assert r["reproducibility"] == "restricted"
    assert "abc1234" in r["reproducibility_note"] and "skillcorner" in r["reproducibility_note"]
    assert len(r["reproducibility_note"]) > 20  # the ADR-067 M4 test's floor
