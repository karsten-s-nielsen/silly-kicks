import math
import os
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

_PUBLIC = "test-token-pining-for-the-data"


def _tmp_db():
    return Path(tempfile.mkdtemp()) / "e2e_stage1.db"


def _two_public_match_ids(provider):
    import scripts._loader_pining as loader

    manifest = loader._list_matches(provider, _PUBLIC, loader._base_url())
    return [m["id"] for m in manifest[:2]]


def _xt_corpus_match_id(provider):
    import scripts._loader_pining as loader

    manifest = loader._list_matches(provider, _PUBLIC, loader._base_url())
    return [manifest[2]["id"]]  # a 3rd match, disjoint from the 2 calibration matches


def _run_stage1(provider, match_ids):
    import scripts._loader_pining as loader
    from scripts.calibrate_tracking_defaults import run_stage

    fold = {}
    for prov, _mid, actions, frames, home in loader.load_matches(
        providers=[provider], match_ids={provider: match_ids}, token=_PUBLIC, tracking_limit=400
    ):
        fold.setdefault(prov, []).append((actions, frames, home))
    result, _objective = run_stage(
        stage=1, fold=fold, n_trials=2, seed=42, store_path=str(_tmp_db()), xt=None, carrier_params=None
    )
    assert result.best is not None
    assert 0.0 <= result.best.metrics["carrier_accuracy"] <= 1.0


def test_stage1_e2e_skillcorner_public():
    # SkillCorner is public on pining; resolve two real ids from the live listing.
    _run_stage1("skillcorner", _two_public_match_ids("skillcorner"))


def test_stage2_e2e_skillcorner_public():
    # M7: the load-bearing CachedObjective (frozen xT + per-fold XGB + invariant/patch split)
    # exercised on REAL data — where H1/H2/NaN/fold-skip surprises actually surface.
    import scripts._loader_pining as loader
    from scripts.calibrate_tracking_defaults import run_stage
    from silly_kicks.calibration._xt import fit_frozen_xt

    ids = _two_public_match_ids("skillcorner") + _xt_corpus_match_id("skillcorner")
    loaded = {
        pid: (a, f, h)
        for _p, pid, a, f, h in loader.load_matches(
            providers=["skillcorner"], match_ids={"skillcorner": ids}, token=_PUBLIC, tracking_limit=600
        )
    }
    calib_ids = ids[:2]
    fold = {"skillcorner": [loaded[i] for i in calib_ids]}
    # Frozen xT fit on a DISJOINT 3rd match (zero overlap with the 2 calibration matches).
    corpus_actions = loaded[ids[2]][0]
    xt = fit_frozen_xt(corpus_actions, exclude_match_ids=set(), match_id_col="game_id", source="e2e")
    result, _objective = run_stage(
        stage=2,
        fold=fold,
        n_trials=2,
        seed=42,
        store_path=str(_tmp_db()),
        xt=xt,  # the FrozenXt artifact (same object the CLI passes); the objective unwraps .xt
        carrier_params={"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0},
    )
    assert result.best is not None
    assert math.isfinite(result.best.metrics["brier"])


@pytest.mark.skipif(not os.environ.get("RUN_HEAVY_E2E"), reason="IDSSE tracking is ~419 MB/match; set RUN_HEAVY_E2E=1")
def test_stage1_e2e_idsse_public():
    # IDSSE is public on pining as of 2026-05-29 (DFL/Sportec XML; heavy download).
    _run_stage1("idsse", _two_public_match_ids("idsse")[:1])
