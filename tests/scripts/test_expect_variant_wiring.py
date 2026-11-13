"""ADR-123 Amendment B: the visibility-gated trainers (xshot/xcross) take the public/owner arm policy
as the neutral ``--arms-json`` parameter and record their corpus identity as a digest, never ids (HFU spec 17.2,
owner 2026-11-09: ``adopt`` rehydrates the canonical public ids off-HF). The
pre-Amendment-B ``--expect-variant`` / ``--providers`` / ``--probe-match-ids-json`` G1 wiring was removed
with the in-job pining path, so those tests are gone; the retained behaviour (ship-label from the arm,
reproducibility, corpus identity, the study enumeration) is exercised here.

ADR-128: both trainers run on the stage contract, so these cases drive the REAL stages in process on the
staged toy corpora (`scripts/_toy_corpora`). The 5c fan-out equality and the study-enumeration CLI
(``--study-list`` / ``--list-studies`` / ``--prep-only``) are gone; their property is the
``tests/scripts/test_parity_x*.py`` gates (every stage output byte-identical at W2N2).
"""

import json

import pytest

pytest.importorskip("xgboost")
pytest.importorskip("ruthless")

from scripts import train_xcross_attempt as xc
from scripts import train_xshot_occurrence as xs

TRAINERS = pytest.mark.parametrize("trainer", [xs, xc], ids=["xshot", "xcross"])


def _toy(tmp_path, trainer):
    """The trainer's staged toy dataset(s) under ``<tmp>/data``; returns ``(data_root, base_args)``."""
    from scripts._toy_corpora import build_xcross_corpus, build_xcross_probe_corpus, build_xshot_corpus

    data = tmp_path / "data"
    if trainer is xs:
        build_xshot_corpus(data / "toy", heavy_episodes=None)
        return data, ["--dataset", "toy"]
    build_xcross_corpus(data / "toy", heavy_episodes=None)
    build_xcross_probe_corpus(data / "probe")
    return data, ["--dataset", "toy", "--probe-dataset", "probe"]


def _keys(data) -> list[tuple[str, str]]:
    """``(source_provider, per-game dir name)`` of every toy unit -- the in-job corpus identity pairs."""
    from scripts._toy_corpora import TOY_GENERATION

    stems = sorted(q.stem for q in (data / "toy" / "shards" / TOY_GENERATION).glob("*.parquet"))
    return [(prov, f"{prov}-{game}") for prov, _, game in (st.partition("__") for st in stems)]


def _metrics(tmp_path, trainer, data, base, arms_json: str) -> dict:
    """Run every stage in process and return the bundle's metrics.json."""
    from tests.scripts._stage_harness import export, run_in_process

    args = [*base, "--arms-json", arms_json, "--n-trials", "2"]
    ctx = run_in_process(trainer.PIPELINE, args, store=tmp_path / "store", data_root=data)
    out = export(ctx, tmp_path / "out")
    return json.loads((out / "metrics.json").read_text(encoding="utf-8"))


@pytest.mark.slow
@TRAINERS
def test_public_run_records_a_digest_and_public_reproducibility(trainer, tmp_path):
    from scripts._corpus_identity import corpus_identity
    from scripts._toy_corpora import xshot_arms

    data, base = _toy(tmp_path, trainer)
    metrics = _metrics(tmp_path, trainer, data, base, xshot_arms(data / "toy", paired=False))
    assert metrics["shipped_variant"] == "public"
    assert "corpus_match_ids" not in metrics and metrics["corpus_n_matches"] == len(_keys(data))
    provs, mids = zip(*sorted(_keys(data)), strict=True)
    want = corpus_identity(provs, mids, all_public=False)
    assert metrics["corpus_match_ids_sha256"] == want["corpus_match_ids_sha256"]
    assert metrics["reproducibility"] == "public"
    # ADR-130: the stoppage record, in metrics.json and in the bundle's metadata.json.
    stoppage = metrics["stoppage"]
    assert stoppage["rule_version"] == 1 and set(stoppage["providers"]) == {prov for prov, _ in _keys(data)}
    meta_name = "metadata.json"
    meta = json.loads(next((tmp_path / "out").rglob(meta_name)).read_text(encoding="utf-8"))
    assert meta["stoppage_provenance"] == stoppage


@pytest.mark.slow
def test_restricted_run_records_a_digest_never_ids(tmp_path):
    """The Hub-job payload (ADR-132, spec 5.5: one sc_extended arm over every game): not paired, so sc_extended
    ships -- restricted -- and metrics.json carries a digest, never a match id."""
    data, base = _toy(tmp_path, xs)
    keys = [f"{prov}__{mid}" for prov, mid in sorted(_keys(data))]
    arms = {"arms": {"sc_extended": keys}, "ship_order": ["sc_extended"]}
    metrics = _metrics(tmp_path, xs, data, base, json.dumps(arms))
    assert metrics["shipped_variant"] == "sc_extended"
    assert "corpus_match_ids" not in metrics and metrics["corpus_n_matches"] == 8
    assert metrics["reproducibility"] == "restricted"
    assert "skillcorner-100" not in (tmp_path / "out" / "metrics.json").read_text(encoding="utf-8")


@TRAINERS
def test_prep_persists_the_inputs_and_enumerates_the_studies(trainer, tmp_path):
    """The ``prep`` stage (the old ``--prep-only``): the --arms policy + persistence + the study enumeration
    are real; only extract and prep run. A paired payload (ADR-132: two arms) yields the 2 x k nested tags + the 2 arm
    studies, then
    one ADR-131 gate-fold study per cv arm and gate fold (``{arm}_gf{k}``, k < cv_arms[arm])."""
    from scripts._stages import ItemStage, LocalStore, make_context, reduce_stage, run_unit, sorted_units
    from scripts._study_shared import load_study_inputs
    from scripts._toy_corpora import xshot_arms
    from tests.scripts._stage_harness import PROV, params_from_args

    data, base = _toy(tmp_path, trainer)
    args = [*base, "--arms-json", xshot_arms(data / "toy", paired=True), "--n-trials", "1"]
    params = params_from_args(trainer.PIPELINE, args)
    ctx = make_context(trainer.PIPELINE, params, PROV, store=LocalStore(tmp_path / "s"), data_root=data)
    for name in ("extract", "prep"):
        st = trainer.PIPELINE.stage(name)
        refs = st.ref_map(ctx) if isinstance(st, ItemStage) else {}
        for key in sorted_units(ctx, st):
            run_unit(ctx, st, key, worker="test", ref=refs.get(key))
        reduce_stage(ctx, name)
    cfg = ctx.upstream("prep")
    assert cfg["run_paired"] is True
    k = 4  # 4 public games -> StratifiedGroupKFold(4) public folds
    want = [f"{n}_f{f}" for f in range(k) for n in ("public", "sc_extended")]
    gf = [f"{n}_gf{g}" for n in sorted(cfg["cv_arms"]) for g in range(cfg["cv_arms"][n])]
    assert sorted(cfg["cv_arms"]) == ["public", "sc_extended"]
    assert cfg["study_tags"] == [*want, "public", "sc_extended", *gf]
    assert sorted_units(ctx, trainer.PIPELINE.stage("study")) == sorted(cfg["study_tags"])
    root, inp = trainer._inputs(ctx)
    assert list(load_study_inputs(root).X.columns) == trainer._feature_names("faithful")
    assert set(inp.arms) == {"public", "sc_extended"}
