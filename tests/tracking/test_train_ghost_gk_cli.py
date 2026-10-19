"""Trainer CLI wires one carrier cp into prepare + fit (PR-S81 / N1 / NP3).

The recorded==used invariant is unit-tested at the model level (test_ghost_gk_r3)
and the prepare level; this guards the place it is actually WIRED -- the trainer
passing one cp to both prepare and fit -- end to end through the CLI.

ADR-128: the CLI is the uniform stage CLI (``--dataset`` over a staged flat dataset, ``--run-all --out``),
and the resume / cache properties are asserted on the stage generations and the per-unit ledger.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts._toy_corpora import build_flat_corpus
from scripts._train_guard import sklearn_supports_training
from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS
from tests.scripts._parity import cli, plan

# VSSOT-IMPL-03: these smokes run the ghost trainer end to end, which now REFUSES to fit under a
# scikit-learn outside the supported range [1.9, 2). Skip (don't fail) when the env is out of range --
# CI's 3.12 primary leg (where these @slow tests run) always resolves sklearn>=1.9, so this never
# reduces real coverage; it only spares a stale dev env (e.g. sklearn 1.8.0) or a future 2.0.
pytestmark = pytest.mark.skipif(
    not sklearn_supports_training(),
    reason="ghost trainer requires scikit-learn in [1.9, 2) (VSSOT-IMPL-03 fit-time guard)",
)

MODULE = "scripts.train_ghost_gk"
# Metrics-only and the slowest step at this scale; nothing asserted here reads it.
BASE = ["--dataset", "tiny", "--n-estimators", "10", "--cv-folds", "2", "--skip-permutation-importance"]


def _tiny_cache(root: Path, n_games: int = 2) -> Path:
    """The staged flat toy (delegates to `_toy_corpora`, ADR-128): games ``100..`` of synthetic ghost
    scenes 1 s apart, one fully-observed provider, home team 1. Returns the ``--data-root``."""
    build_flat_corpus(root / "tiny", n_games=n_games, providers=("gradientsports",), n_frames=39, dt=1.0)
    return root


def _invoke(data: Path, store: Path, out: Path, *extra: str) -> None:
    """One CLI run against an EXPLICIT data/store/out triple, so a test can invoke it more than once."""
    cli(MODULE, store, data, [*BASE, *extra], "--run-all", "--out", str(out))


def _gen(data: Path, store: Path, stage: str, *extra: str) -> Path:
    stages = {s["name"]: s for s in plan(MODULE, store, data, [*BASE, *extra])["stages"]}
    return store / stages[stage]["generation"]


def _attempts(generation: Path) -> dict[str, int]:
    """Per unit key, how many times it has RUN (a resumed unit is never run again)."""
    recs = (json.loads(p.read_text(encoding="utf-8")) for p in generation.glob("*.ledger.json"))
    return {r["key"]: r["attempt"] for r in recs}


def _run(tmp_path: Path, *extra: str) -> dict:
    data = _tiny_cache(tmp_path / "data")
    out = tmp_path / "out"
    _invoke(data, tmp_path / "store", out, *extra)
    return json.loads((out / "metadata.json").read_text())


@pytest.mark.slow
def test_cli_records_supplied_carrier_params(tmp_path):
    meta = _run(tmp_path, "--carrier-beta", "0.9", "--carrier-gamma", "0.3", "--carrier-tolerance", "2.5")
    assert meta["carrier_params"] == {"tolerance_m": 2.5, "beta": 0.9, "gamma": 0.3}
    assert meta["version"] == "1.3.0"  # Option A artifact format (gk_y ensemble + baselines)
    assert meta["serve_estimator"] == "boosted_mean"


@pytest.mark.slow
def test_cli_omitted_records_library_default(tmp_path):
    meta = _run(tmp_path)
    assert meta["carrier_params"] == dict(DEFAULT_CARRIER_PARAMS)


@pytest.mark.slow
def test_cli_feature_set_position_only(tmp_path):
    # `--feature-set position_only` trains a 21-feature ghost (the 5 velocity features dropped) end to
    # end through the trainer harness, the parity gate, and the position_only chirality /
    # feature-contract guards. Exercises the trainer subprocess glue (CLI -> stage token ->
    # prepare -> fit -> save) the model round-trips cannot.
    from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES_POSITION_ONLY, GhostGkModel

    meta = _run(tmp_path, "--feature-set", "position_only")
    assert meta["feature_set"] == "position_only"
    assert meta["feature_names"] == list(GHOST_GK_FEATURE_NAMES_POSITION_ONLY)
    assert len(meta["feature_names"]) == 21  # the 5 velocity features dropped
    # Loads through the position_only chirality + feature-contract guards:
    GhostGkModel.load(tmp_path / "out")


# ---------------------------------------------------------------------------
# ADR-052 / ADR-128: the per-game extraction is an ItemStage, and every extraction input keys its token
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_a_changed_extraction_PARAMETER_misses_the_feature_cache(tmp_path):
    """Both directions, because only the pair is evidence.

    The recorded cache token used to be the penalty-area geometry alone, so a re-run at a different
    ``--subsample-fps`` reused the previous run's feature matrix while ``metadata.json`` recorded
    the NEW parameters -- the recorded==used invariant (PR-S81) broken by the cache beneath it.
    Asserting only the MISS would pass equally well on a token that never hits at all, which would
    replace a silent-staleness bug with a silent-recompute one.
    """
    data, store = _tiny_cache(tmp_path / "data"), tmp_path / "store"
    _invoke(data, store, tmp_path / "o1", "--subsample-fps", "25.0")
    extract25 = _gen(data, store, "extract", "--subsample-fps", "25.0")
    cache25 = _gen(data, store, "cache", "--subsample-fps", "25.0")
    first = (_attempts(extract25), _attempts(cache25))
    assert first[0] and set(first[0].values()) == {1}

    _invoke(data, store, tmp_path / "o2", "--subsample-fps", "25.0")
    assert (_attempts(extract25), _attempts(cache25)) == first, "identical parameters must hit (resume)"

    extract125 = _gen(data, store, "extract", "--subsample-fps", "12.5")
    assert extract125 != extract25, "a changed --subsample-fps silently reused the previous features"
    _invoke(data, store, tmp_path / "o3", "--subsample-fps", "12.5")
    assert set(_attempts(extract125).values()) == {1}, "the changed parameter did not re-extract"


@pytest.mark.slow
def test_a_RESUMED_extraction_skips_games_that_already_have_a_shard(tmp_path):
    """Drop every stage DOWNSTREAM of the extraction. The run must find every game already done -- and
    produce the same training matrix it produced the first time."""
    import shutil

    data, store = _tiny_cache(tmp_path / "data"), tmp_path / "store"
    _invoke(data, store, tmp_path / "o1")
    extract = _gen(data, store, "extract")
    before = _attempts(extract)
    assert before and set(before.values()) == {1}
    n_first = json.loads((tmp_path / "o1" / "metrics.json").read_text())["n_samples"]

    for stage in ("cache", "cv", "fit", "importance", "assemble"):
        shutil.rmtree(store / "ghost_gk" / stage)
    _invoke(data, store, tmp_path / "o2")
    assert _attempts(extract) == before, "the shards were ignored and every game was re-extracted"
    assert json.loads((tmp_path / "o2" / "metrics.json").read_text())["n_samples"] == n_first

    # Non-vacuity: with the shards ALSO gone, the same invocation must do the work again. Without
    # this half, a run that silently produced no shards at all would look like a healthy resume.
    for stage in ("cache", "cv", "fit", "importance", "assemble"):
        shutil.rmtree(store / "ghost_gk" / stage)
    for key in before:
        (extract / f"{key}.parquet").unlink()
    _invoke(data, store, tmp_path / "o3")
    assert set(_attempts(extract).values()) == {2}, "the resume does not depend on the shards"


@pytest.mark.slow
def test_a_game_with_NO_home_mapping_is_skipped_and_counted(tmp_path):
    """A unit whose game is absent from the home-team map is an EXCLUSION: it leaves no shard, but the
    stage summary REPORTS it with its reason -- and it never reaches the training set."""
    data = _tiny_cache(tmp_path / "data", n_games=3)
    # Map two of the three games; the third has no home_team_id and must drop out.
    (data / "tiny" / "home_teams.json").write_text(json.dumps({"100": 1, "101": 1}), encoding="utf-8")
    store = tmp_path / "store"
    _invoke(data, store, tmp_path / "out")

    summary = json.loads((_gen(data, store, "extract") / "_summary.json").read_text(encoding="utf-8"))
    assert summary["excluded"] == {"gradientsports__102": "no home_team_id in mapping"}
    assert summary["n_done"] == 2
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert metrics["n_games"] == 2, "a game with no home mapping still reached the training set"


@pytest.mark.slow
def test_a_CORRECTED_home_team_mapping_reaches_the_features(tmp_path):
    """`home_team_id` drives the goal-relative flip of every feature and label in a shard, so a
    corrected mapping must reach the features.

    The failure it closes was silent and total: the shard was skipped, the correction never landed,
    and the whole-corpus cache accepted the result too, so the model trained on wrong-handed data with
    nothing anywhere reporting a problem. ADR-128 (plan Task 13 M3): the mapping is part of the staged
    dataset, and the dataset identity keys the extract token, so a corrected mapping is a NEW extract
    generation -- here a same-SIZE edit, which a names-and-sizes listing alone would miss.
    """
    data, store = _tiny_cache(tmp_path / "data"), tmp_path / "store"
    ht = data / "tiny" / "home_teams.json"
    ht.write_text(json.dumps({"100": 1, "101": 1}), encoding="utf-8")
    _invoke(data, store, tmp_path / "o1")
    first = _gen(data, store, "extract")

    ht.write_text(json.dumps({"100": 2, "101": 1}), encoding="utf-8")  # same byte size, corrected game 100
    corrected = _gen(data, store, "extract")
    assert corrected != first, "the corrected mapping did not change the extract generation"
    _invoke(data, store, tmp_path / "o2")
    # The corrected game was RE-EXTRACTED under the new mapping (a fresh unit in the new generation), never
    # skipped as an existing shard. (The toy scene's labelled attack direction makes the two shards'
    # bytes coincide, so the evidence is the re-extraction itself, as the legacy shard-key assertion was.)
    assert _attempts(corrected) == {"gradientsports__100": 1, "gradientsports__101": 1}
    assert _attempts(first) == {"gradientsports__100": 1, "gradientsports__101": 1}, "the old generation was rerun"
