"""TF-61 driver smoke gates (AST/text; no network). Provenance wiring for train/evolve is covered by
`test_provenance_wiring.py` (both are enrolled in ARTIFACT_DRIVERS); these add the parser-safety +
publish-seam checks that gate doesn't."""

import pathlib

_SCRIPTS = pathlib.Path(__file__).parents[2] / "scripts"


def _src(name: str) -> str:
    return (_SCRIPTS / f"{name}.py").read_text(encoding="utf-8")


def test_train_and_evolve_have_parsers_and_provenance():
    # a real argparse parser -> `--help` is safe (parserless scripts run main() on --help, memory).
    for name in ("train_xsuccess", "evolve_xsuccess_features"):
        s = _src(name)
        assert "add_argument" in s, f"{name}: no argparse parser"
        assert "require_clean_tree" in s, f"{name}: no clean-tree guard"
        assert "for_each" in s, f"{name}: not adopting the _driver corpus seam (ADR-052)"
        assert "rev-parse" not in s, f"{name}: shells out to git rev-parse"


def test_train_and_evolve_are_public_corpus_gated_by_the_open_data_loader():
    # TF61-IMPL-01: the redistributability gate for these open-data-only trainers IS the open-data
    # source (`open_data_source`, the Task 8.5 refs+load factory over `list_open_data_refs` /
    # `load_open_data_match`; open-data == public by construction — the train_pass_completion
    # convention; `assert_public_corpus` is a pining-corpus visibility check, circular for open data).
    # The invariant restored here: the trainer loads ONLY open data and NEVER the pining loader,
    # so a bundled artifact cannot train on non-redistributable data.
    for name in ("train_xsuccess", "evolve_xsuccess_features"):
        s = _src(name)
        assert "open_data_source" in s, f"{name}: not sourced from the public open-data loader"
        assert "_loader_pining" not in s, f"{name}: imports the pining loader (non-redistributable)"


def test_publish_uses_card_required_seam():
    s = _src("publish_xsuccess")
    assert "add_argument" in s
    assert "publish_model_with_card" in s, "publish must use the ADR-088 card-required seam"
    assert "upload_model_only" not in s, "publish must NOT call upload_model_only directly (ADR-088)"
    assert "--model-card" in s and "REQUIRED" in s, "publish must require a model card"


def test_train_shard_schema_has_no_end_columns():
    # END-BLIND: the fit shard must not carry the realized end (target leakage, spec §5.2).
    s = _src("train_xsuccess")
    assert '"end_x"' not in s and '"end_y"' not in s, "train_xsuccess shard leaks the realized end"
