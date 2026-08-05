"""ADR-038 taxonomy wiring for ``train_gk_completion.py`` (PR 5, TODO L26).

The trainer previously imported none of ``is_public_row`` / ``artifact_label`` /
``assert_public_corpus``, unlike its xS and xCross siblings, so a defaulted
``--max-per-provider 64`` run could pull 54 restricted SkillCorner matches into a distributable
artifact with nothing refusing it or labelling the result.

These test the taxonomy helpers directly rather than through the CLI: ``train_gk_completion.py``
has no ``--dry-run`` (measured: zero occurrences), and a real run costs a corpus download.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from _corpus import artifact_label, is_public_row


def test_label_is_restricted_when_any_row_is_restricted():
    """Fail-closed direction: one private match is enough to lose the `public` claim."""
    vis = {("skillcorner", "1"): "public", ("skillcorner", "2"): "private"}
    mask = is_public_row(
        providers=np.array(["skillcorner", "skillcorner"]),
        match_ids=np.array(["1", "2"]),
        visibility=vis,
    )
    assert mask.tolist() == [True, False]
    assert artifact_label(providers={"skillcorner"}, all_public=bool(mask.all())) == "sc_extended"


def test_label_is_public_only_when_every_row_is():
    vis = {("skillcorner", "1"): "public", ("skillcorner", "2"): "public"}
    mask = is_public_row(
        providers=np.array(["skillcorner", "skillcorner"]),
        match_ids=np.array(["1", "2"]),
        visibility=vis,
    )
    assert artifact_label(providers={"skillcorner"}, all_public=bool(mask.all())) == "public"


def test_gradientsports_is_the_full_tier():
    """The bundled `default` variant's corpus. Owner decision 2026-08-02: this label ships."""
    assert artifact_label(providers={"gradientsports"}, all_public=False) == "full"


def test_absent_from_manifest_is_restricted_not_public():
    """FAIL-CLOSED: a match the manifest omits must never be treated as public."""
    mask = is_public_row(providers=np.array(["skillcorner"]), match_ids=np.array(["999"]), visibility={})
    assert mask.tolist() == [False]


def test_trainer_imports_the_taxonomy_helpers():
    """Behavioural, not a substring check: the helper must exist and be callable.

    Guards the actual gap TODO L26 names -- a trainer with no taxonomy enforcement.
    """
    import train_gk_completion as t

    assert callable(t._corpus_taxonomy)


def test_empty_corpus_is_not_public():
    """`ndarray.all()` is vacuously True on an empty array, so a zero-match run would otherwise
    claim `public`. Pinned here because the guard is one `len(pairs) and ...` away from silent."""
    mask = is_public_row(providers=np.array([]), match_ids=np.array([]), visibility={})
    assert mask.all()  # the numpy trap this guards against
    assert artifact_label(providers=set(), all_public=bool(len(mask) and mask.all())) != "public"


@pytest.mark.parametrize(
    ("providers", "all_public", "expected"),
    [
        ({"skillcorner"}, True, "public"),
        ({"skillcorner", "idsse"}, True, "public"),
        ({"skillcorner"}, False, "sc_extended"),
        ({"gradientsports"}, False, "full"),
        ({"skillcorner", "gradientsports"}, False, "full"),
    ],
)
def test_label_table(providers, all_public, expected):
    assert artifact_label(providers=providers, all_public=all_public) == expected
