"""G1 wiring: the public-arm trainers refuse a restricted corpus BEFORE extraction, refuse to ship a
variant other than the expected one (also via the persisted config an --assemble worker reads), and
record their corpus identity -- ids only when all-public (spec section 5)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("ruthless")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import _loader_pining  # the module object the trainers' function-local imports resolve

from scripts import train_xcross_attempt as xc
from scripts import train_xshot_occurrence as xs


def _listing(monkeypatch, vis):
    monkeypatch.setattr(_loader_pining, "select_match_ids", lambda **kw: sorted(vis))
    monkeypatch.setattr(_loader_pining, "match_visibility", lambda provs, **kw: dict(vis))

    def _no_extraction(*a, **k):
        raise AssertionError("extraction started -- the G1 preflight must refuse first")

    monkeypatch.setattr(_loader_pining, "pining_source", _no_extraction)


_ARGV = ["--providers", "skillcorner", "--output-dir", "{out}", "--expect-variant", "public", "--allow-dirty"]


@pytest.mark.parametrize("trainer", [xs, xc], ids=["xshot", "xcross"])
def test_expect_public_refuses_a_restricted_request_before_extraction(trainer, tmp_path, monkeypatch):
    _listing(monkeypatch, {("skillcorner", "1886347"): "public", ("skillcorner", "900"): "private"})
    with pytest.raises(SystemExit, match="non-public"):
        trainer.main([a.format(out=tmp_path) for a in _ARGV])


@pytest.mark.parametrize("trainer", [xs, xc], ids=["xshot", "xcross"])
def test_expect_public_passes_an_all_public_request(trainer, tmp_path, monkeypatch):
    """The other side of the band: an all-public request clears the preflight and reaches extraction."""
    _listing(monkeypatch, {("skillcorner", "1886347"): "public"})
    with pytest.raises(AssertionError, match="extraction started"):
        trainer.main([a.format(out=tmp_path) for a in _ARGV])


def _inputs(root, feature_names, *, public: bool, expect, extra_config, n_games=4, rows=60):
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
            "run_paired": False,
            "run_prov": {"commit": "test", "dirty": False, "tree_state": "clean"},
            "expect_variant": expect,
            "objective_inputs": {"driver": "test_expect_variant_wiring"},  # D21 identity the prep persists
            **extra_config,
        },
    )


def _names(trainer):
    if trainer is xs:
        from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL

        return XSHOT_FEATURE_NAMES_FAITHFUL, {}, {}
    from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL

    return XCROSS_FEATURE_NAMES_FAITHFUL, {"ship_variant": None}, {"run_probe": False}


@pytest.mark.parametrize("trainer", [xs, xc], ids=["xshot", "xcross"])
def test_ship_time_check_refuses_from_the_persisted_config_before_any_fit(trainer, tmp_path):
    """An --assemble worker reads the expectation from the persisted config: a restricted single-candidate
    corpus (owner-tier SkillCorner only) would ship sc_extended, so expect=public refuses with no fit."""
    names, extra, kw = _names(trainer)
    _inputs(tmp_path, names, public=False, expect="public", extra_config=extra)
    with pytest.raises(SystemExit, match="would ship 'sc_extended'"):
        trainer.assemble_studies(tmp_path, study_shard_dir=tmp_path, **kw)
    assert not (tmp_path / "art" / "model.json").exists()


@pytest.mark.slow
@pytest.mark.parametrize("trainer", [xs, xc], ids=["xshot", "xcross"])
def test_public_run_records_ids_and_public_reproducibility(trainer, tmp_path):
    names, extra, kw = _names(trainer)
    _inputs(tmp_path, names, public=True, expect="public", extra_config=extra, rows=80)
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

    _inputs(tmp_path, XSHOT_FEATURE_NAMES_FAITHFUL, public=False, expect=None, extra_config={}, rows=80)
    metrics, _model = xs.assemble_studies(tmp_path, study_shard_dir=tmp_path)
    assert metrics["shipped_variant"] == "sc_extended"
    assert "corpus_match_ids" not in metrics and metrics["corpus_n_matches"] == 4
    assert metrics["reproducibility"] == "restricted"
    assert "1886347" not in (tmp_path / "art" / "metrics.json").read_text(encoding="utf-8")


def test_probe_only_matches_never_enter_training(tmp_path, monkeypatch):
    """D5(c): GS probe matches are extracted in a separate pass for the TF-19 probe; training sees only the
    public allowlist, and the probe sample lists exactly the probe matches (held out by construction)."""
    import json

    from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL

    names = XCROSS_FEATURE_NAMES_FAITHFUL
    vis = {("skillcorner", "1886347"): "public", ("idsse", "DFL-MAT-J03WMX"): "public"}
    monkeypatch.setattr(_loader_pining, "select_match_ids", lambda **kw: sorted(vis))
    monkeypatch.setattr(_loader_pining, "match_visibility", lambda provs, **kw: dict(vis))
    monkeypatch.setattr(
        _loader_pining,
        "pining_source",
        lambda provs, match_ids=None, **kw: ([(p, m) for p in provs for m in (match_ids or {}).get(p, [])], None),
    )

    def fake_extract(
        source, horizon, *, shard_root, probe_providers, probe_comparison_providers, feature_set, load, **kw
    ):
        cohort, rows = xc._new_probe_cohort(), list(source)
        for prov, mid in rows:
            if prov in probe_providers:
                cohort["frames"].append(pd.DataFrame({"game_id": [mid], "frame_id": [1]}))
                cohort["actions"].append(pd.DataFrame({"game_id": [mid]}))
                cohort["home"] = 1
                cohort["matches"].append([prov, mid])
                cohort["match_groups"][mid] = [mid]
        X = pd.DataFrame(0.0, index=range(len(rows)), columns=names)
        mids = np.array([m for _, m in rows], dtype=object)
        provs = np.array([p for p, _ in rows], dtype=object)
        return X, np.zeros(len(rows), int), mids.copy(), provs, mids, (cohort, xc._new_probe_cohort(), 0)

    monkeypatch.setattr(xc, "_extract", fake_extract)
    import scripts._study_shared as ss

    seen = {}

    def stop(root, **kw):
        seen["providers"] = set(kw["providers"].tolist())
        raise SystemExit("stop")

    monkeypatch.setattr(ss, "persist_study_inputs", stop)
    allow, probe = tmp_path / "allow.json", tmp_path / "probe.json"
    allow.write_text(json.dumps({"skillcorner": ["1886347"], "idsse": ["DFL-MAT-J03WMX"]}))
    probe.write_text(json.dumps({"gradientsports": ["10502", "10503"]}))
    argv = [
        "--providers",
        "idsse,skillcorner",
        "--match-ids-json",
        str(allow),
        "--max-per-provider",
        "10",
        "--output-dir",
        str(tmp_path / "o"),
        "--expect-variant",
        "public",
        "--probe-providers",
        "gradientsports",
        "--probe-comparison-providers",
        "",
        "--probe-match-ids-json",
        str(probe),
        "--allow-dirty",
    ]
    with pytest.raises(SystemExit, match="stop"):
        xc.main(argv)
    assert seen["providers"] == {"idsse", "skillcorner"}  # the probe matches never reach training
    meta = json.loads((tmp_path / "o" / "xcross_attempt_v1" / "_probe_sample" / "meta.json").read_text())
    assert meta["probe_matches"] == [["gradientsports", "10502"], ["gradientsports", "10503"]]


def _forbid_pining(monkeypatch):
    """Hermetic (B r5 CCC-PLAN-42): if a refusal regresses, main() continues into the corpus listing. That
    must fail the test, never reach the live pining API (the round-4 incident class)."""

    def _boom(*a, **k):
        raise AssertionError("a unit test reached the pining API")

    for name in ("select_match_ids", "match_visibility", "pining_source", "list_match_refs", "load_match"):
        monkeypatch.setattr(_loader_pining, name, _boom)


def test_probe_match_ids_must_name_a_probe_provider(tmp_path, capsys, monkeypatch):
    """Asserts the MESSAGE: a bare SystemExit also passes on RED, where argparse rejects the unknown flag
    (B r4 CCC-PLAN-39)."""
    import json

    _forbid_pining(monkeypatch)
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({"gradientsports": ["10502"]}))
    with pytest.raises(SystemExit):
        xc.main(
            [
                "--providers",
                "idsse",
                "--output-dir",
                str(tmp_path / "o"),
                "--probe-providers",
                "skillcorner",
                "--probe-comparison-providers",
                "",
                "--probe-match-ids-json",
                str(probe),
                "--allow-dirty",
            ]
        )
    assert "must all be in --probe-providers" in capsys.readouterr().err


def test_a_probe_match_the_training_corpus_can_load_is_refused(tmp_path, capsys, monkeypatch):
    """D5(c) held-out by construction: a probe id that training could also load is refused up front, not left
    to the DGX acceptance check (B r4 CCC-PLAN-39)."""
    import json

    _forbid_pining(monkeypatch)
    probe, allow = tmp_path / "probe.json", tmp_path / "allow.json"
    probe.write_text(json.dumps({"skillcorner": ["1886347"]}))
    allow.write_text(json.dumps({"skillcorner": ["1886347"], "idsse": ["DFL-MAT-J03WMX"]}))
    with pytest.raises(SystemExit):
        xc.main(
            [
                "--providers",
                "idsse,skillcorner",
                "--match-ids-json",
                str(allow),
                "--output-dir",
                str(tmp_path / "o"),
                "--probe-providers",
                "skillcorner",
                "--probe-comparison-providers",
                "",
                "--probe-match-ids-json",
                str(probe),
                "--allow-dirty",
            ]
        )
    err = capsys.readouterr().err
    assert "disjoint from training" in err and "1886347" not in err  # the refusal names a count, never an id


@pytest.mark.parametrize("meta", [None, {"probe_matches": [["gradientsports", "99999"]]}], ids=["absent", "stale"])
def test_a_missing_or_stale_cached_probe_sample_is_refused(tmp_path, monkeypatch, meta):
    """Step 3(c): on a feature-cache hit, the persisted probe sample must match --probe-match-ids-json. An
    ABSENT one is refused too: it used to escape as FileNotFoundError (B r5, outside its round)."""
    import json

    import _cache

    _forbid_pining(monkeypatch)
    monkeypatch.setattr(xc, "_corpus_fingerprint", lambda args: "fp")
    monkeypatch.setattr(_cache, "cache_is_valid", lambda cache_dir, fingerprint: True)
    cache = tmp_path / "o" / "xcross_attempt_v1" / "_feature_cache"
    cache.mkdir(parents=True)
    pd.DataFrame({"a": [0.0]}).to_parquet(cache / "features.parquet")
    # Match how the trainer READS the cache (B r6 CCC-PLAN-44): labels are numeric and loaded without
    # allow_pickle (`train_xcross_attempt.py:880`); groups / providers / match_ids are object arrays loaded
    # with allow_pickle=True. An object-dtype labels.npy would die in np.load before the probe check.
    np.save(cache / "labels.npy", np.array([0]))
    for name in ("groups", "providers", "match_ids"):
        np.save(cache / f"{name}.npy", np.array(["x"], dtype=object), allow_pickle=True)
    if meta is not None:
        (cache.parent / "_probe_sample").mkdir()
        (cache.parent / "_probe_sample" / "meta.json").write_text(json.dumps(meta))
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({"gradientsports": ["10502", "10503"]}))
    with pytest.raises(SystemExit, match="_probe_sample"):
        xc.main(
            [
                "--providers",
                "idsse,skillcorner",
                "--output-dir",
                str(tmp_path / "o"),
                "--probe-providers",
                "gradientsports",
                "--probe-comparison-providers",
                "",
                "--probe-match-ids-json",
                str(probe),
                "--allow-dirty",
            ]
        )


@pytest.mark.slow
@pytest.mark.parametrize("trainer", [xs, xc], ids=["xshot", "xcross"])
def test_study_list_fan_out_equals_the_serial_assemble(trainer, tmp_path):
    """D3 (launcher Task 9 in miniature): --list-studies + --study-list workers + --assemble == serial assemble,
    byte-identical booster -- the CLI path the launcher's f1b mode drives."""
    import json

    from scripts._study_shared import persist_study_inputs
    from tests.scripts import test_train_study_parallel_parity as P

    corpus = P._synthetic_paired_corpus_xcross if trainer is xc else P._synthetic_paired_corpus
    X, y, groups, providers, match_ids, is_public = corpus()

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
                "objective_inputs": {"driver": "test_expect_variant_wiring"},  # D21 identity the prep persists
            },
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
    """D3 (B r3 CCC-PLAN-27): the launcher's f1b mode starts from --prep-only, so a defect here must fail CI,
    not the DGX. Extraction is faked with the existing PAIRED study fixture (a public-only corpus enumerates
    no studies, which would make the check vacuous); visibility, persistence and enumeration are real."""
    import json

    from scripts._study_shared import load_study_inputs
    from tests.scripts import test_train_study_parallel_parity as P

    X, y, groups, providers, match_ids, is_public = (
        P._synthetic_paired_corpus_xcross if trainer is xc else P._synthetic_paired_corpus
    )()
    pairs = sorted({(str(p), str(m), bool(pub)) for p, m, pub in zip(providers, match_ids, is_public, strict=True)})
    vis = {(p, m): ("public" if pub else "private") for p, m, pub in pairs}
    allow: dict[str, list[str]] = {}
    for p, m, _pub in pairs:
        allow.setdefault(p, []).append(m)
    monkeypatch.setattr(_loader_pining, "select_match_ids", lambda **kw: sorted(vis))
    monkeypatch.setattr(_loader_pining, "match_visibility", lambda provs, **kw: dict(vis))
    monkeypatch.setattr(
        _loader_pining,
        "pining_source",
        lambda provs, match_ids=None, **kw: ([(p, m) for p in provs for m in (match_ids or {}).get(p, [])], None),
    )

    def fake_extract(source, horizon, **kw):
        base = (X, y, groups, providers, match_ids)
        return (*base, (xc._new_probe_cohort(), xc._new_probe_cohort(), 0)) if trainer is xc else base

    monkeypatch.setattr(trainer, "_extract", fake_extract)
    monkeypatch.setattr(trainer, "assemble_studies", lambda *a, **k: pytest.fail("--prep-only must not assemble"))
    # The fixture's "public" ids (m0, m1, m2) are synthetic, so the real PUBLIC_CORPUS registry check refuses
    # them ("UNREGISTERED public match(es)", B r4 CCC-PLAN-38). That check is not under test here (Task 2 tests
    # G1 and the registry); stub ONLY it, on the module object the trainers' function-local import resolves.
    import _corpus  # the scripts/ module object, as `_loader_pining` above

    monkeypatch.setattr(_corpus, "assert_public_corpus", lambda *a, **k: None)
    allow_file = tmp_path / "allow.json"
    allow_file.write_text(json.dumps(allow))
    trainer.main(
        [
            "--providers",
            ",".join(sorted(allow)),
            "--match-ids-json",
            str(allow_file),
            "--max-per-provider",
            "10",
            "--n-trials",
            "1",
            "--output-dir",
            str(tmp_path / "o"),
            "--allow-dirty",
            "--prep-only",
        ]
    )
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    root = Path(out["study_root"])
    assert root.name == "studies" and (tmp_path / "o") in root.parents
    assert out["studies"]  # paired corpus -> a real fan-out list, not the vacuous []
    assert out["studies"] == trainer.enumerate_studies(root)
    assert list(load_study_inputs(root).X.columns) == list(X.columns)


@pytest.mark.parametrize("trainer", [xs, xc], ids=["xshot", "xcross"])
def test_list_studies_prints_the_tags(trainer, tmp_path, capsys):
    import json

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
    )
    trainer.main(["--shard-root", str(tmp_path), "--list-studies"])
    assert json.loads(capsys.readouterr().out) == trainer.enumerate_studies(tmp_path)
