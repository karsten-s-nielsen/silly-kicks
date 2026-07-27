"""Trainer-side guards (ADR-050): feature-cache identity, corpus fingerprints, provider fail-fast.

These live in ``scripts/``, which has NO ``__init__.py`` -- so trainers are loaded by file path and
their own imports use the bare ``from _cache import ...`` form.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None, f"could not load scripts/{name}.py"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ghost_cache_token_is_derived_from_the_geometry_constants():
    """A hand-bumped literal goes stale INSIDE the re-fit cycle it protects: extract features, flip
    the constant, re-run -- and the second run silently reuses the first run's 40.3 features while
    stamping a 20.16 contract. Deriving the token from the constants auto-invalidates on the flip
    with zero discipline required."""
    import silly_kicks.tracking._ghost_gk as gg

    t = _load("train_ghost_gk")
    before = t.cache_token()
    original = gg._PENALTY_AREA_Y_MIN
    try:
        gg._PENALTY_AREA_Y_MIN = (68.0 - 40.32) / 2.0
        assert t.cache_token() != before, "token must change when the box constant changes"
    finally:
        gg._PENALTY_AREA_Y_MIN = original


def test_corpus_fingerprint_distinguishes_corpora():
    """The whole point: a changed corpus must MISS. The constant 'schema-v2' token cannot."""
    # NOTE the import form: `scripts/` has NO __init__.py, so `from scripts._cache import ...` is a
    # ModuleNotFoundError. This is the established idiom (tests/scripts/test_cache_schema.py).
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    from _cache import corpus_fingerprint

    a = corpus_fingerprint([("gs", "1", "public"), ("gs", "2", "public")])
    b = corpus_fingerprint([("gs", "1", "public")])
    c = corpus_fingerprint([("gs", "2", "public"), ("gs", "1", "public")])  # order-insensitive
    assert a != b
    assert a == c


def test_trainers_no_longer_gate_on_a_constant_token():
    """Asserted on the ASSIGNMENT, not on the string appearing anywhere in the file.

    A bare ``'"schema-v2"' not in src`` also forbids *explaining* the change in a docstring, which
    is the opposite of what we want: the reason a guard was replaced is exactly the thing worth
    recording next to it. The defect was the module-level constant being handed to
    ``cache_is_valid``; that is what this checks.
    """
    import ast

    for name in ("train_xshot_occurrence", "train_xcross_attempt"):
        path = REPO / "scripts" / f"{name}.py"
        src = path.read_text(encoding="utf-8")
        assigned = {
            t.id
            for node in ast.parse(src).body
            if isinstance(node, ast.Assign)
            for t in node.targets
            if isinstance(t, ast.Name)
        }
        assert "_CACHE_FINGERPRINT" not in assigned, (
            f"{name} still defines a module-level constant fingerprint; the cache gate must be a live per-corpus hash"
        )
        assert "corpus_fingerprint" in src, f"{name} must build a live fingerprint"
        assert "select_match_ids" in src, (
            f"{name} must key the fingerprint on the REQUESTED corpus via the shared selection "
            f"helper, not on a second copy of the allowlist/cap rule"
        )


def test_selection_is_single_sourced_between_fingerprint_and_extraction():
    """``select_match_ids`` and ``load_matches`` must apply the SAME allowlist/cap rule. Two copies
    would let the fingerprint describe a corpus the extraction never loaded -- the cache would then
    be validated against a fiction."""
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    from _loader_pining import _wanted_for_provider

    manifest = ["a", "b", "c", "d"]
    assert _wanted_for_provider(manifest, "gs", None, None) == ["a", "b", "c", "d"]
    assert _wanted_for_provider(manifest, "gs", None, 2) == ["a", "b"]
    assert _wanted_for_provider(manifest, "gs", {"gs": ["c", "d"]}, None) == ["c", "d"]
    assert _wanted_for_provider(manifest, "gs", {"gs": ["c", "d"]}, 1) == ["c"]
    # a provider absent from the allowlist falls back to the full manifest, not to empty
    assert _wanted_for_provider(manifest, "sk", {"gs": ["c"]}, None) == ["a", "b", "c", "d"]


def test_unclassified_provider_fails_BEFORE_any_fitting(monkeypatch):
    """Spy the fit: without this the test passes on the pre-existing mid-run raise and proves
    nothing. The value of this change is entirely in WHEN it fires."""
    import pytest as _pytest

    import silly_kicks.tracking._ghost_gk as gg

    calls = {"n": 0}
    monkeypatch.setattr(gg.GhostGkModel, "fit", lambda self, *a, **k: calls.__setitem__("n", 1))

    t = _load("train_ghost_gk")
    with _pytest.raises(ValueError, match="provider"):
        t.validate_corpus_providers(["gradientsports", "not_a_provider"])
    assert calls["n"] == 0


def test_validate_provider_is_shared_not_duplicated():
    """Two copies of the membership set drift the moment a provider is added -- silently, because
    an unclassified provider only surfaces deep inside a run."""
    import pytest as _pytest

    import silly_kicks.tracking._ghost_gk as gg

    assert callable(gg.validate_provider)
    gg.validate_provider("gradientsports")
    with _pytest.raises(ValueError, match="provider"):
        gg.validate_provider("nope")


def test_keeper_detection_mask_still_rejects_an_unknown_provider():
    """The extraction of `validate_provider` must be behaviour-preserving at the original site --
    moving a check earlier must not mean removing it from where it already was."""
    import pandas as pd
    import pytest as _pytest

    from silly_kicks.tracking._ghost_gk import keeper_detection_mask

    with _pytest.raises(ValueError, match="provider"):
        keeper_detection_mask(pd.Series([True, False]), provider="nope")
