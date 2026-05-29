import numpy as np
import pandas as pd
import pytest

from silly_kicks.calibration._xt import fit_frozen_xt, load_xt, save_xt


def _toy_actions(match_ids, n_per=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for mid in match_ids:
        for i in range(n_per):
            rows.append(
                {
                    "game_id": mid,
                    "action_id": i,
                    "period_id": 1,
                    "time_seconds": float(i),
                    "team_id": 1,
                    "player_id": 10,
                    "start_x": rng.uniform(0, 105),
                    "start_y": rng.uniform(0, 68),
                    "end_x": rng.uniform(0, 105),
                    "end_y": rng.uniform(0, 68),
                    "type_id": 0,
                    "result_id": 1,
                    "bodypart_id": 0,
                }
            )
    return pd.DataFrame(rows)


def test_fit_excludes_calibration_matches_and_records_provenance():
    corpus = _toy_actions(["c1", "c2", "c3"])  # game_id is the match key here
    frozen = fit_frozen_xt(corpus, exclude_match_ids={"c3"}, match_id_col="game_id", source="test-corpus")
    assert "c3" not in frozen.corpus_match_ids
    assert frozen.corpus_match_ids == ("c1", "c2")
    assert frozen.source == "test-corpus"
    assert frozen.grid_shape == frozen.xt.xT.shape
    assert len(frozen.sha256) == 64


def test_fit_raises_when_no_corpus_remains_after_exclusion():
    corpus = _toy_actions(["c1"])
    with pytest.raises(ValueError, match="disjoint corpus is empty"):
        fit_frozen_xt(corpus, exclude_match_ids={"c1"}, match_id_col="game_id", source="x")


def test_fit_fails_closed_when_excluded_id_absent_from_corpus():
    # H2: an excluded id that doesn't exist in the corpus means the exclusion no-ops -> would LEAK.
    corpus = _toy_actions(["c1", "c2", "c3"])
    with pytest.raises(ValueError, match="were NOT found in corpus"):
        # 'pining-99' is a different id space than corpus game_ids => must fail closed.
        fit_frozen_xt(corpus, exclude_match_ids={"c1", "pining-99"}, match_id_col="game_id", source="x")


def test_fit_records_n_excluded():
    corpus = _toy_actions(["c1", "c2", "c3"])
    frozen = fit_frozen_xt(corpus, exclude_match_ids={"c1", "c2"}, match_id_col="game_id", source="x")
    assert frozen.n_excluded == 2
    assert frozen.manifest()["n_excluded"] == 2


def test_save_load_roundtrip_preserves_grid_and_sha(tmp_path):
    corpus = _toy_actions(["c1", "c2"])
    frozen = fit_frozen_xt(corpus, exclude_match_ids=set(), match_id_col="game_id", source="x")
    path = tmp_path / "xt.npz"
    save_xt(frozen, path)
    loaded = load_xt(path)
    assert loaded.sha256 == frozen.sha256
    np.testing.assert_array_equal(loaded.xt.xT, frozen.xt.xT)
    assert loaded.corpus_match_ids == frozen.corpus_match_ids


def test_load_detects_tampered_grid(tmp_path):
    corpus = _toy_actions(["c1", "c2"])
    frozen = fit_frozen_xt(corpus, exclude_match_ids=set(), match_id_col="game_id", source="x")
    path = tmp_path / "xt.npz"
    save_xt(frozen, path)
    # Tamper: rewrite the grid but keep the stored sha256
    data = dict(np.load(path, allow_pickle=True))
    data["xT"] = data["xT"] + 1.0
    np.savez(path, **data)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_xt(path)
