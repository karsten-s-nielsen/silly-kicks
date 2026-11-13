"""The licensing-taxonomy helpers (spec 3.2, ADR-132): the raw-redistributable pin (`RAW_REDISTRIBUTABLE`),
the per-row `is_public_row` mask, and the `artifact_label` provenance fold. A model trained on licensed data
must NEVER be labelled `public`; that end-to-end control now lives launcher-side and is tested in
`tests/scripts/hf_jobs/test_arm_policy.py` (see the ADR-123 Amendment B note below).

`scripts/` is on sys.path via tests/scripts/conftest.py, so `_corpus` imports by bare name.
"""

import numpy as np
from _corpus import RAW_REDISTRIBUTABLE, artifact_label, is_public_row


def test_absent_visibility_is_restricted():
    """FAIL-CLOSED: unknown provenance is never public."""
    vis = {("skillcorner", "1886347"): "public"}
    got = is_public_row(
        providers=np.array(["skillcorner", "skillcorner"]),
        match_ids=np.array(["1886347", "9999999"]),  # second is absent from the map
        visibility=vis,
    )
    assert list(got) == [True, False]


def test_a_restricted_skillcorner_match_is_not_public():
    vis = {("skillcorner", "1021404"): "private"}
    got = is_public_row(providers=np.array(["skillcorner"]), match_ids=np.array(["1021404"]), visibility=vis)
    assert list(got) == [False]


def test_label_is_never_public_when_the_ship_mask_contains_restricted_rows():
    """The bug that shipped: providers={skillcorner, idsse}, no GS -> old code said "public". ADR-132: the
    label is provenance only; a ship mask with a licensed row is `sc_extended` whatever its providers (`full`
    is retired)."""
    assert artifact_label(all_obtainable=False) == "sc_extended"
    assert artifact_label(all_obtainable=True) == "public"


def test_public_corpus_is_the_known_27():
    # SkillCorner grew 10 -> 20 on 2026-09-09 (a second public drop); IDSSE unchanged at 7 -> 27 total.
    assert len(RAW_REDISTRIBUTABLE["skillcorner"]) == 20
    assert len(RAW_REDISTRIBUTABLE["idsse"]) == 7


def test_public_corpus_skillcorner_matches_the_sample_loader():
    # The 20 public SkillCorner ids are LICENSING-critical and duplicated in
    # tests/_skillcorner_sample.MATCH_IDS (the sample loader). Guard the two literals against drift:
    # a divergence could load-as-sample a match not registered-as-public, or the reverse.
    from tests._skillcorner_sample import MATCH_IDS

    assert RAW_REDISTRIBUTABLE["skillcorner"] == frozenset(MATCH_IDS)


# NOTE (ADR-123 Amendment B): the obsolete in-job `--providers` taxonomy test
# (`test_a_restricted_corpus_NEVER_ships_a_public_label`) and its helpers were removed here --
# the trainer no longer computes visibility; the launcher does. "A restricted corpus never ships a
# public label" is now enforced + tested in `tests/scripts/hf_jobs/test_arm_policy.py`
# (`test_owner_skillcorner_forms_sc_extended_not_public` + the unregistered->not-public assertions).


def test_the_corpus_pin_and_the_visibility_arm_are_orthogonal():
    """Task 13: the --match-ids-json pin says WHICH matches load; visibility says WHICH ARM they
    join. They are orthogonal -- a pinned PRIVATE match is still private (never enters the public
    arm just because it was explicitly selected to load)."""
    vis = {("skillcorner", "1021404"): "private"}
    got = is_public_row(providers=np.array(["skillcorner"]), match_ids=np.array(["1021404"]), visibility=vis)
    assert list(got) == [False]
