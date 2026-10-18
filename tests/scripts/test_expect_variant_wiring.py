"""ADR-123 Amendment B: the visibility-gated trainers (xshot/xcross) take the public/owner arm policy
as the neutral ``--arms`` parameter and record their corpus identity -- ids only when all-public. The
pre-Amendment-B ``--expect-variant`` / ``--providers`` / ``--probe-match-ids-json`` G1 wiring was removed
with the in-job pining path, so those tests are gone; the retained behaviour (ship-label from the arm,
reproducibility, corpus identity, prep-only + study fan-out) is exercised here on the ``--arms`` API.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("ruthless")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from scripts import train_xcross_attempt as xc
from scripts import train_xshot_occurrence as xs


def _arms_from(providers: np.ndarray, is_public: np.ndarray) -> dict:
    """Build the candidate-arm masks a launcher would pass -- the pre-Amendment-B ``_build_candidate_masks``
    logic, now the TEST's job (the trainer only applies the provided memberships)."""
    is_sc_private = (providers == "skillcorner") & ~is_public
    return {"public": is_public, "sc_extended": is_public | is_sc_private, "full": np.ones(len(providers), bool)}


def _single_arm(name: str, n: int) -> dict:
    return {name: np.ones(n, bool)}


def _inputs(
    root, feature_names, *, shipped: str, public: bool, run_paired: bool, extra_config, arms, n_games=4, rows=60
):
    from scripts._study_shared import persist_study_inputs

    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.standard_normal((n_games * rows, len(feature_names))), columns=feature_names)
    y = (X.iloc[:, 0] > 0).to_numpy().astype(int)
    mids = np.repeat(["1886347", "1899585", "DFL-MAT-J03WMX", "DFL-MAT-J03WN1"][:n_games], rows)
    persist_study_inputs(
        root,
        X=X,
        y=y,
        groups=mids.copy(),
        providers=np.repeat(["skillcorner", "skillcorner", "idsse", "idsse"][:n_games], rows),
        match_ids=mids,
        is_public=np.full(n_games * rows, public),
        config={
            "n_trials": 1,
            "negative_subsample": None,
            "seed": 42,
            "feature_set": "faithful",
            "horizon_seconds": 1.0,
            "study_db_dir": str(root),
            "artifact_dir": str(root / "art"),
            "run_paired": run_paired,
            "run_prov": {"commit": "test", "dirty": False, "tree_state": "clean"},
            "expect_variant": None,
            "ship_order": [shipped],  # single-candidate: the shipped label is the arm name from the param
            "objective_inputs": {"driver": "test_expect_variant_wiring"},
            **extra_config,
        },
        arms=arms,
    )


def _names(trainer):
    if trainer is xs:
        from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL

        return XSHOT_FEATURE_NAMES_FAITHFUL, {}, {}
    from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL

    return XCROSS_FEATURE_NAMES_FAITHFUL, {"ship_variant": None}, {"run_probe": False}


@pytest.mark.slow
@pytest.mark.parametrize("trainer", [xs, xc], ids=["xshot", "xcross"])
def test_public_run_records_ids_and_public_reproducibility(trainer, tmp_path):
    names, extra, kw = _names(trainer)
    _inputs(
        tmp_path,
        names,
        shipped="public",
        public=True,
        run_paired=False,
        extra_config=extra,
        arms=_single_arm("public", 4 * 80),
        rows=80,
    )
    metrics, _model = trainer.assemble_studies(tmp_path, study_shard_dir=tmp_path, **kw)
    assert metrics["shipped_variant"] == "public"
    assert metrics["corpus_match_ids"] == [
        ["idsse", "DFL-MAT-J03WMX"],
        ["idsse", "DFL-MAT-J03WN1"],
        ["skillcorner", "1886347"],
        ["skillcorner", "1899585"],
    ]
    assert metrics["reproducibility"] == "public"


@pytest.mark.slow
def test_restricted_run_records_a_digest_never_ids(tmp_path):
    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL

    _inputs(
        tmp_path,
        XSHOT_FEATURE_NAMES_FAITHFUL,
        shipped="sc_extended",
        public=False,
        run_paired=False,
        extra_config={},
        arms=_single_arm("sc_extended", 4 * 80),
        rows=80,
    )
    metrics, _model = xs.assemble_studies(tmp_path, study_shard_dir=tmp_path)
    assert metrics["shipped_variant"] == "sc_extended"
    assert "corpus_match_ids" not in metrics and metrics["corpus_n_matches"] == 4
    assert metrics["reproducibility"] == "restricted"
    assert "1886347" not in (tmp_path / "art" / "metrics.json").read_text(encoding="utf-8")


@pytest.mark.slow
@pytest.mark.parametrize("trainer", [xs, xc], ids=["xshot", "xcross"])
def test_study_list_fan_out_equals_the_serial_assemble(trainer, tmp_path):
    """--list-studies + --study-list workers + --assemble == serial assemble, byte-identical booster."""
    from scripts._study_shared import persist_study_inputs
    from tests.scripts import test_train_study_parallel_parity as P

    corpus = P._synthetic_paired_corpus_xcross if trainer is xc else P._synthetic_paired_corpus
    X, y, groups, providers, match_ids, is_public = corpus()
    arms = _arms_from(providers, is_public)

    def _persist(root):
        persist_study_inputs(
            root,
            X=X,
            y=y,
            groups=groups,
            providers=providers,
            match_ids=match_ids,
            is_public=is_public,
            config={
                "n_trials": 2,
                "negative_subsample": None,
                "seed": 42,
                "feature_set": "faithful",
                "horizon_seconds": 1.0,
                "study_db_dir": str(root),
                "artifact_dir": str(root / "art"),
                "run_paired": True,
                "run_prov": {"commit": "t", "dirty": False, "tree_state": "clean"},
                "ship_variant": None,
                "expect_variant": None,
                "ship_order": ["public", "sc_extended", "full"],
                "objective_inputs": {"driver": "test_expect_variant_wiring"},
            },
            arms=arms,
        )

    kw = {"run_probe": False} if trainer is xc else {}
    serial, fan = tmp_path / "serial", tmp_path / "fan"
    _persist(serial)
    _persist(fan)
    _m, model_serial = trainer.assemble_studies(serial, study_shard_dir=serial, **kw)
    tags = trainer.enumerate_studies(fan)
    for i, half in enumerate((tags[::2], tags[1::2])):
        f = tmp_path / f"tags{i}.json"
        f.write_text(json.dumps(half))
        trainer.main(["--shard-root", str(fan), "--study-list", str(f)])
    _m2, model_fan = trainer.assemble_studies(fan, study_shard_dir=fan, **kw)
    assert model_serial._booster.save_raw("json") == model_fan._booster.save_raw("json")


@pytest.mark.parametrize("trainer", [xs, xc], ids=["xshot", "xcross"])
def test_prep_only_persists_the_inputs_prints_the_studies_and_never_assembles(trainer, tmp_path, monkeypatch, capsys):
    """The launcher's f1b mode starts from --prep-only; a defect here must fail CI, not the DGX. Extraction
    is faked with the PAIRED study fixture; the --arms policy + persistence + enumeration are real."""
    from scripts._study_shared import load_study_inputs
    from tests.scripts import test_train_study_parallel_parity as P

    X, y, groups, providers, match_ids, is_public = (
        P._synthetic_paired_corpus_xcross if trainer is xc else P._synthetic_paired_corpus
    )()
    # the launcher's --arms payload for this corpus. The fake _extract bypasses repackage, so the in-job
    # match_key is match_key(source_provider, raw match_id) == "{provider}__{match_id}" here.
    keys = [f"{p}__{m}" for p, m in zip(providers, match_ids, strict=True)]
    arms = {
        "public": sorted({k for k, pub in zip(keys, is_public, strict=True) if pub}),
        "sc_extended": sorted(
            {k for k, p, pub in zip(keys, providers, is_public, strict=True) if pub or p == "skillcorner"}
        ),
        "full": sorted(set(keys)),
    }
    arms_file = tmp_path / "arms.json"
    arms_file.write_text(json.dumps({"arms": arms, "ship_order": ["public", "sc_extended", "full"]}))

    def fake_extract(source, horizon, **kw):
        list(source)  # drain the --data-dir generator
        base = (X, y, groups, providers, match_ids)
        return (*base, (xc._new_probe_cohort(), xc._new_probe_cohort(), 0)) if trainer is xc else base

    monkeypatch.setattr(trainer, "_extract", fake_extract)
    monkeypatch.setattr(trainer, "_iter_matches_from_dir", lambda *a, **k: iter([]))
    monkeypatch.setattr(trainer, "assemble_studies", lambda *a, **k: pytest.fail("--prep-only must not assemble"))
    data_dir = tmp_path / "corpus"
    (data_dir / "g0").mkdir(parents=True)  # a non-empty dir so the --data-dir fingerprint has content
    argv = [
        "--data-dir",
        str(data_dir),
        "--arms",
        str(arms_file),
        "--n-trials",
        "1",
        "--output-dir",
        str(tmp_path / "o"),
        "--allow-dirty",
        "--prep-only",
    ]
    if trainer is xc:
        probe = tmp_path / "probe"
        (probe / "g").mkdir(parents=True)
        argv += ["--probe-data-dir", str(probe)]
        monkeypatch.setattr(xc, "_write_probe_sample", lambda *a, **k: None)
    trainer.main(argv)
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    root = Path(out["study_root"])
    assert root.name == "studies" and (tmp_path / "o") in root.parents
    assert out["studies"]  # paired corpus -> a real fan-out list, not the vacuous []
    assert out["studies"] == trainer.enumerate_studies(root)
    assert list(load_study_inputs(root).X.columns) == list(X.columns)


@pytest.mark.parametrize("trainer", [xs, xc], ids=["xshot", "xcross"])
def test_list_studies_prints_the_tags(trainer, tmp_path, capsys):
    from scripts._study_shared import persist_study_inputs
    from tests.scripts import test_train_study_parallel_parity as P

    corpus = P._synthetic_paired_corpus_xcross if trainer is xc else P._synthetic_paired_corpus
    X, y, groups, providers, match_ids, is_public = corpus()
    persist_study_inputs(
        tmp_path,
        X=X,
        y=y,
        groups=groups,
        providers=providers,
        match_ids=match_ids,
        is_public=is_public,
        config={"run_paired": True, "n_trials": 1, "negative_subsample": None, "seed": 42},
        arms=_arms_from(providers, is_public),
    )
    trainer.main(["--shard-root", str(tmp_path), "--list-studies"])
    assert json.loads(capsys.readouterr().out) == trainer.enumerate_studies(tmp_path)
