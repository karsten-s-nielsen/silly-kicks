"""TF-61 driver smoke gates (AST/text; no network). Provenance wiring for train/evolve is covered by
`test_provenance_wiring.py` (both are enrolled in ARTIFACT_DRIVERS); these add the parser-safety +
publish-seam checks that gate doesn't."""

import pathlib

_SCRIPTS = pathlib.Path(__file__).parents[2] / "scripts"


def _src(name: str) -> str:
    return (_SCRIPTS / f"{name}.py").read_text(encoding="utf-8")


def test_train_and_evolve_have_parsers_and_provenance():
    # a real parser -> `--help` is safe (parserless scripts run main() on --help, memory). `train_xsuccess` is a
    # stage-contract pipeline (HFU R3-10): the uniform CLI (`cli_main`) is its parser and the stage framework
    # stamps its provenance (`ctx.prov`, refused on a dirty tree by the campaign); `evolve` keeps its own driver.
    s = _src("train_xsuccess")
    assert "PIPELINE = Pipeline(" in s and "cli_main(PIPELINE)" in s, "train_xsuccess: not a stage-contract pipeline"
    assert 'prov["commit"]' in s, "train_xsuccess: does not stamp training_commit from the run provenance"
    assert "rev-parse" not in s, "train_xsuccess: shells out to git rev-parse"
    s = _src("evolve_xsuccess_features")
    assert "add_argument" in s, "evolve_xsuccess_features: no argparse parser"
    assert "require_clean_tree" in s, "evolve_xsuccess_features: no clean-tree guard"
    assert "for_each" in s, "evolve_xsuccess_features: not adopting the _driver corpus seam (ADR-052)"
    assert "rev-parse" not in s, "evolve_xsuccess_features: shells out to git rev-parse"


def test_train_and_evolve_are_public_corpus_gated_by_the_open_data_loader():
    # TF61-IMPL-01: the redistributability gate for these open-data-only trainers IS the open-data
    # source (`list_open_data_refs` / `load_open_data_match`, directly or via the `open_data_source`
    # factory; open-data == public by construction — the train_pass_completion convention;
    # `assert_public_corpus` is a pining-corpus visibility check, circular for open data).
    # The invariant restored here: the trainer loads ONLY open data and NEVER the pining loader,
    # so a bundled artifact cannot train on non-redistributable data.
    s = _src("train_xsuccess")
    assert "list_open_data_refs" in s and "load_open_data_match" in s, "train_xsuccess: not open-data sourced"
    assert "_loader_pining" not in s, "train_xsuccess: imports the pining loader (non-redistributable)"
    s = _src("evolve_xsuccess_features")
    assert "open_data_source" in s, "evolve_xsuccess_features: not sourced from the public open-data loader"
    assert "_loader_pining" not in s, "evolve_xsuccess_features: imports the pining loader (non-redistributable)"


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
