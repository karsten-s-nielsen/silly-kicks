"""Task 6 guard: an F1b study run on the shared-mmap corpus == the in-memory study, byte-identical."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("ruthless")

from scripts import train_xshot_occurrence as tr
from scripts._corpus_mmap import load_design_matrix, persist_design_matrix


@pytest.mark.slow
def test_study_from_shared_corpus_matches_in_memory_booster_bytes(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((400, 3)).astype(np.float64), columns=["a", "b", "c"])
    y = (X["a"] > 0).to_numpy().astype(np.int8)
    groups = np.arange(400) % 6

    s_params, s_booster = tr._fit_study_for_test(X, y, groups, "full_f0", n_trials=5, seed=42)

    p = tmp_path / "dm"
    persist_design_matrix(X, y, groups, p)
    Xs, ys, gs = load_design_matrix(p)
    p_params, p_booster = tr._fit_study_for_test(Xs, ys, gs, "full_f0", n_trials=5, seed=42)

    assert s_params == p_params
    assert s_booster.save_raw("json") == p_booster.save_raw("json")  # byte-identical model
    assert s_booster.feature_names == p_booster.feature_names == ["a", "b", "c"]


def _synthetic_paired_corpus():
    """A small paired corpus (public + owner SkillCorner + Gradient Sports) with the real 27 faithful
    feature columns, a strong signal so the acceptance gates pass, and enough public games for k folds."""
    from silly_kicks.tracking._xshot_occurrence import XSHOT_FEATURE_NAMES_FAITHFUL

    rng = np.random.default_rng(7)
    # (provider, is_public) per game; >=3 public games -> StratifiedGroupKFold(3); GS -> run_paired.
    games = [
        ("skillcorner", True),
        ("skillcorner", True),
        ("idsse", True),
        ("skillcorner", False),
        ("skillcorner", False),  # owner-tier SkillCorner -> sc_extended
        ("gradientsports", False),
        ("gradientsports", False),  # owner GS -> full != sc_extended
    ]
    rows_per_game = 60
    Xs, ys, groups, provs, pubs, mids = [], [], [], [], [], []
    for gi, (prov, pub) in enumerate(games):
        block = rng.standard_normal((rows_per_game, len(XSHOT_FEATURE_NAMES_FAITHFUL)))
        Xs.append(block)
        ys.append((block[:, 0] > 0).astype(int))  # strong signal on feature 0 -> gates pass
        groups.append(np.full(rows_per_game, f"g{gi}"))
        provs.append(np.full(rows_per_game, prov))
        pubs.append(np.full(rows_per_game, pub))
        mids.append(np.full(rows_per_game, f"m{gi}"))
    X = pd.DataFrame(np.vstack(Xs), columns=XSHOT_FEATURE_NAMES_FAITHFUL)
    return (
        X,
        np.concatenate(ys).astype(int),
        np.concatenate(groups),
        np.concatenate(provs),
        np.concatenate(mids),
        np.concatenate(pubs).astype(bool),
    )


@pytest.mark.slow
def test_parallel_assemble_equals_serial_weights_byte_identical(tmp_path):
    X, y, groups, providers, match_ids, is_public = _synthetic_paired_corpus()
    from scripts._study_shared import persist_study_inputs

    def _config(art):
        return {
            "n_trials": 2,
            "negative_subsample": None,
            "seed": 42,
            "feature_set": "faithful",
            "horizon_seconds": 1.0,
            "study_db_dir": str(art),
            "artifact_dir": str(art / "art"),
            "run_paired": True,
            "run_prov": {"commit": "test", "dirty": False, "tree_state": "clean"},
        }

    # Serial: studies computed inline during assemble (empty cache).
    root_s = tmp_path / "serial"
    persist_study_inputs(
        root_s,
        X=X,
        y=y,
        groups=groups,
        providers=providers,
        match_ids=match_ids,
        is_public=is_public,
        config=_config(root_s),
    )
    m_serial, model_serial = tr.assemble_studies(root_s, study_shard_dir=root_s)

    # Parallel: each study run as a separate worker, THEN assemble reads the cache.
    root_p = tmp_path / "parallel"
    persist_study_inputs(
        root_p,
        X=X,
        y=y,
        groups=groups,
        providers=providers,
        match_ids=match_ids,
        is_public=is_public,
        config=_config(root_p),
    )
    tags = tr.enumerate_studies(root_p)
    assert len(tags) == 9  # 3 candidates x 3 public folds
    for tag in tags:
        tr.run_one_study(root_p, tag)
    m_parallel, model_parallel = tr.assemble_studies(root_p, study_shard_dir=root_p)

    assert m_serial["shipped_variant"] == m_parallel["shipped_variant"]  # same ship verdict
    sb, pb = model_serial._booster, model_parallel._booster
    assert sb is not None and pb is not None  # fitted by assemble_studies
    assert sb.save_raw("json") == pb.save_raw("json")


def _synthetic_paired_corpus_xcross():
    """A paired corpus with the real 16 faithful xcross columns; score_differential kept in-range so the
    B6 guard passes; a strong signal so the acceptance gates pass."""
    from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL

    rng = np.random.default_rng(11)
    games = [
        ("skillcorner", True),
        ("skillcorner", True),
        ("idsse", True),
        ("skillcorner", False),
        ("skillcorner", False),
        ("gradientsports", False),
        ("gradientsports", False),
    ]
    rows_per_game = 60
    cols = XCROSS_FEATURE_NAMES_FAITHFUL
    Xs, ys, groups, provs, pubs, mids = [], [], [], [], [], []
    for gi, (prov, pub) in enumerate(games):
        block = rng.standard_normal((rows_per_game, len(cols)))
        block[:, cols.index("score_differential")] = rng.integers(-3, 4, rows_per_game)  # in-range for B6
        Xs.append(block)
        ys.append((block[:, 0] > 0).astype(int))  # strong signal on feature 0 -> gates pass
        groups.append(np.full(rows_per_game, f"g{gi}"))
        provs.append(np.full(rows_per_game, prov))
        pubs.append(np.full(rows_per_game, pub))
        mids.append(np.full(rows_per_game, f"m{gi}"))
    X = pd.DataFrame(np.vstack(Xs), columns=cols)
    return (
        X,
        np.concatenate(ys).astype(int),
        np.concatenate(groups),
        np.concatenate(provs),
        np.concatenate(mids),
        np.concatenate(pubs).astype(bool),
    )


@pytest.mark.slow
def test_xcross_parallel_assemble_equals_serial_weights_byte_identical(tmp_path):
    from scripts import train_xcross_attempt as trx
    from scripts._study_shared import persist_study_inputs

    X, y, groups, providers, match_ids, is_public = _synthetic_paired_corpus_xcross()

    def _config(art):
        return {
            "n_trials": 2,
            "negative_subsample": None,
            "seed": 42,
            "feature_set": "faithful",
            "horizon_seconds": 1.0,
            "study_db_dir": str(art),
            "artifact_dir": str(art / "art"),
            "run_paired": True,
            "ship_variant": None,
            "run_prov": {"commit": "test", "dirty": False, "tree_state": "clean"},
        }

    root_s = tmp_path / "serial"
    persist_study_inputs(
        root_s,
        X=X,
        y=y,
        groups=groups,
        providers=providers,
        match_ids=match_ids,
        is_public=is_public,
        config=_config(root_s),
    )
    m_serial, model_serial = trx.assemble_studies(root_s, study_shard_dir=root_s, run_probe=False)

    root_p = tmp_path / "parallel"
    persist_study_inputs(
        root_p,
        X=X,
        y=y,
        groups=groups,
        providers=providers,
        match_ids=match_ids,
        is_public=is_public,
        config=_config(root_p),
    )
    tags = trx.enumerate_studies(root_p)
    assert len(tags) == 9  # 3 candidates x 3 public folds
    for tag in tags:
        trx.run_one_study(root_p, tag)
    m_parallel, model_parallel = trx.assemble_studies(root_p, study_shard_dir=root_p, run_probe=False)

    assert m_serial["shipped_variant"] == m_parallel["shipped_variant"]
    sb, pb = model_serial._booster, model_parallel._booster
    assert sb is not None and pb is not None  # fitted by assemble_studies
    assert sb.save_raw("json") == pb.save_raw("json")
