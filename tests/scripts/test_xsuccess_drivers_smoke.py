"""TF-61 driver smoke gates (AST/text; no network). Provenance wiring for train/evolve is covered by
`test_provenance_wiring.py` (both are enrolled in ARTIFACT_DRIVERS); these add the parser-safety +
publish-seam checks that gate doesn't."""

import pathlib

_SCRIPTS = pathlib.Path(__file__).parents[2] / "scripts"


def _src(name: str) -> str:
    return (_SCRIPTS / f"{name}.py").read_text(encoding="utf-8")


def test_train_is_a_pipeline_and_evolve_has_a_parser_and_provenance():
    # train_xsuccess is an ADR-128 pipeline (HFU spec 15.5a): its CLI and clean-tree guard are `_stages.main`'s.
    t = _src("train_xsuccess")
    assert "PIPELINE = Pipeline(" in t and "raise SystemExit(main(PIPELINE))" in t
    # a real argparse parser -> `--help` is safe (parserless scripts run main() on --help, memory).
    for name in ("evolve_xsuccess_features",):
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
    sources = {"train_xsuccess": "_open_data_stages", "evolve_xsuccess_features": "open_data_source"}
    for name, source in sources.items():
        s = _src(name)
        assert source in s, f"{name}: not sourced from the public open-data loader"
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


def _patch_open_data(monkeypatch, comps):
    """The open-data seams the pipeline calls: 2 matches per release, ids ``c*1000+s+{1,2}``, each with ~60
    on-ball rows of mixed types and outcomes (the xsuccess shard columns). No network."""
    import numpy as np
    import pandas as pd

    import scripts._sb_open_data as sod
    from scripts._driver_sources import LoadedMatch
    from silly_kicks.spadl import config as spadlconfig

    types = [spadlconfig.actiontype_id[t] for t in ("pass", "dribble", "cross", "shot")]

    def _actions(gid):
        rng, n = np.random.default_rng(gid), 60
        return pd.DataFrame({
            "game_id": gid, "period_id": 1, "time_seconds": np.arange(n, dtype=float),
            "type_id": rng.choice(types, n), "result_id": rng.integers(0, 2, n), "bodypart_id": 0,
            "start_x": rng.uniform(0, 105, n), "start_y": rng.uniform(0, 68, n),
        })  # fmt: skip

    monkeypatch.setattr(sod, "assert_statsbomb_open_data_mode", lambda: None)
    monkeypatch.setattr(sod, "all_open_competitions", lambda: list(comps))
    monkeypatch.setattr(
        sod,
        "list_open_data_refs",
        lambda cs, *, match_ids=None, max_matches=None: [
            sod.OpenDataRef(c, s, str(c * 1000 + s + k), 1) for c, s in cs for k in (1, 2)
        ],
    )
    monkeypatch.setattr(
        sod,
        "load_open_data_match",
        lambda ref, *, preserve_native=(), cache_dir=None: LoadedMatch(
            "statsbomb", ref.match_id, _actions(int(ref.match_id)), pd.DataFrame(), 1, None, None
        ),
    )


def test_xsuccess_metrics_records_the_open_data_identity(tmp_path, monkeypatch):
    """HFU spec 14.3: metrics.json records the corpus identity of the refs the run trained on (the lock's
    ``sb_open.identity.open`` function) and ``hpo_trials``; verify compares the identity with the lock."""
    import json

    import scripts.train_xsuccess as T
    from scripts._corpus_identity import open_data_identity
    from tests.scripts._stage_harness import export, run_in_process

    comps = [(43, 106), (11, 90), (55, 43), (2, 27)]
    _patch_open_data(monkeypatch, comps)
    ctx = run_in_process(T.PIPELINE, ["--all-competitions"], store=tmp_path / "s", data_root=None)
    out = export(ctx, tmp_path / "o")
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    want = open_data_identity([str(c * 1000 + s + k) for c, s in comps for k in (1, 2)])
    assert {k: metrics[k] for k in want} == want
    assert metrics["hpo_trials"] == 0
    # Decision P4 (ADR-128), as win_probability and the legacy fit: metrics.json records the extract pass.
    passed = {"generation": ctx.tokens["extract"], "n_attempted": 8, "n_failed": 0, "n_counters_unrecorded": 0}
    assert {k: metrics[k] for k in (*passed, "n_excluded")} == {**passed, "n_excluded": 0}


def test_hpo_stage_plans_the_long_job_limit_and_the_fold_pool(tmp_path):
    """HFU spec 15.7: the real xsuccess plan gives ``hpo`` 14,400 s and a 5-wide InnerPool (``_job_spec`` reads
    ``timeout_s`` from this record, ``test_job_spec_takes_a_stage_timeout_from_the_real_plan``)."""
    import scripts.train_xsuccess as T
    from scripts._stages import LocalStore, make_context, plan
    from tests.scripts._stage_harness import PROV, params_from_args

    ctx = make_context(T.PIPELINE, params_from_args(T.PIPELINE, []), PROV, store=LocalStore(tmp_path), data_root=None)
    rec = {r["name"]: r for r in plan(ctx)["stages"]}
    assert {k: rec["hpo"][k] for k in ("timeout_s", "inner_parallel", "inner_width")} == {
        "timeout_s": 14400,
        "inner_parallel": True,
        "inner_width": 5,
    }
    assert {rec[n]["timeout_s"] for n in ("refs", "extract")} == {None}


def test_card_states_corpus_identity():
    """HFU spec 14.13: the card states the corpus identity verify checks (count + digest)."""
    from scripts.train_xsuccess import render_model_card

    card = render_model_card({"corpus_match_ids_sha256": "ab" * 32, "corpus_n_matches": 3961})
    assert f"Corpus identity: 3961 matches, sha256 {'ab' * 32}" in card


def test_fit_plans_36_units_without_an_inner_pool(tmp_path):
    """HFU fit-units spec 3.2: the `fit` stage is a fixed grid of 36 units (one xgboost fit each), with no
    InnerPool and the default job limit; the deterministic byte proof is `test_xsuccess_fit_units.py`."""
    import scripts.train_xsuccess as T
    from scripts._stages import LocalStore, make_context, plan
    from tests.scripts._stage_harness import PROV, params_from_args

    ctx = make_context(T.PIPELINE, params_from_args(T.PIPELINE, []), PROV, store=LocalStore(tmp_path), data_root=None)
    rec = {r["name"]: r for r in plan(ctx)["stages"]}["fit"]
    assert rec["n_units"] == 36 and rec["units"] == list(T.FIT_UNITS)
    assert (rec["inner_parallel"], rec["timeout_s"]) == (False, None)
    assert not hasattr(T, "_fit_map") and not hasattr(T, "cross_val_metrics")
