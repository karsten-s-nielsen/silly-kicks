"""Task 2 (adopt-for-refit seam): the trainers stamp corpus identity in the DIGEST form unconditionally.

The in-job trainer is visibility-blind (ADR-123 Amendment B); it must record the
`corpus_match_ids_sha256`/`corpus_n_matches` digest, never the exact `corpus_match_ids` (an owner id must
never reach a copied metrics.json; `adopt` rehydrates the exact PUBLIC ids off-HF). So `_assemble_work`
must call `corpus_identity(..., all_public=False)` -- never `all_public=is_public.all()`, which stamps the
exact form for an all-public arm and then mismatches the digest the staging record + expectation check use.
Guarded at the source of each trainer's `_assemble_work` (a behavioural fit test is slow/e2e)."""

from __future__ import annotations

import inspect

import scripts.train_xcross_attempt as xc
import scripts.train_xshot_occurrence as xs


def test_xshot_stamp_is_digest_always():
    src = inspect.getsource(xs._assemble_work)
    assert "corpus_identity(" in src
    assert "all_public=False" in src
    assert "is_public.all()" not in src  # must NOT feed visibility into the stamp form


def test_xcross_stamp_is_digest_always():
    src = inspect.getsource(xc._assemble_work)
    assert "corpus_identity(" in src
    assert "all_public=False" in src
    assert "is_public.all()" not in src


def test_corpus_identity_library_fn_unchanged():
    # only the CALL SITE flips; the library still returns the exact form when asked for all_public
    from scripts._corpus_identity import corpus_identity

    exact = corpus_identity(["idsse"], ["m1"], all_public=True)
    assert "corpus_match_ids" in exact and "corpus_match_ids_sha256" not in exact
    digest = corpus_identity(["idsse"], ["m1"], all_public=False)
    assert "corpus_match_ids_sha256" in digest and "corpus_match_ids" not in digest
