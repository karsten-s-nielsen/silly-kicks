import ast


def test_corpus_identity_has_no_visibility_symbols():
    src = open("scripts/_corpus_identity.py", encoding="utf-8").read()
    for banned in ("PUBLIC_CORPUS", "is_public_row", "assert_public_corpus", "artifact_label", "match_visibility"):
        assert banned not in src, banned


def test_corpus_identity_imports_no_corpus():
    # the relocation is pointless if _corpus_identity re-pulls _corpus into the closure
    tree = ast.parse(open("scripts/_corpus_identity.py", encoding="utf-8").read())
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m == "_corpus" or m.endswith("._corpus") for m in mods), mods


def test_corpus_identity_exports_the_neutral_helpers():
    import scripts._corpus_identity as ci

    for fn in (
        "corpus_identity",
        "reproducibility",
        "check_shipped_variant",
        "check_expected_variant",
        "match_id_pairs",
        "canonical_providers",
    ):
        assert hasattr(ci, fn), fn


def test_canonical_providers_normalizes_converter_to_identity():
    # provider_list records the dataset IDENTITY, not the SPADL converter name: the DFL/Bundesliga matches
    # carry source_provider="sportec" but their identity is "idsse" (the vocabulary the metadata test,
    # cards and consumers expect). Sorted + unique; non-converter providers pass through unchanged.
    from scripts._corpus_identity import canonical_providers

    assert canonical_providers(["sportec", "skillcorner", "skillcorner", "sportec"]) == ["idsse", "skillcorner"]
    assert canonical_providers(["skillcorner"]) == ["skillcorner"]
    assert canonical_providers(["sportec"]) == ["idsse"]


def test_invert_injective_builds_the_inverse_and_refuses_a_collision():
    # B27(a): _IDENTITY_TO_CONVERTER is the inverse of _CONVERTER_TO_IDENTITY; a two-converters-on-one-identity
    # map has no well-defined inverse and must raise at build time, else the arm key for the losing converter
    # would silently resolve to the wrong in-job match key (the DFL-drop class).
    import pytest

    from scripts._corpus_identity import _CONVERTER_TO_IDENTITY, _IDENTITY_TO_CONVERTER, _invert_injective

    assert _invert_injective({"sportec": "idsse"}) == {"idsse": "sportec"}
    assert _IDENTITY_TO_CONVERTER == {v: k for k, v in _CONVERTER_TO_IDENTITY.items()}
    with pytest.raises(ValueError, match="not injective"):
        _invert_injective({"sportec": "idsse", "wyscout": "idsse"})  # two converters -> one identity
