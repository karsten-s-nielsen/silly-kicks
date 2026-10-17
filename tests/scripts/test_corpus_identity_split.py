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
