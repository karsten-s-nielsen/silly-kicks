"""TF-60 PR3 Task 7: trainer glue for the sweeper re-fit (>30 m stratum + coverage + grid wiring).

Component-level (non-slow): the metric helpers, the CLI arg, `grid_spec_from_args`, and the
BEHAVIORAL stale-shard-token proof (`grid_x_max` keys the shard generation). PLUS one `@slow`
end-to-end stage-CLI smoke (ADR-128) (`test_trainer_main_smoke_threads_grid_and_metrics`) that fits a real model
and is therefore SKIPPED where sklearn is outside the supported fit range [1.9, 2) -- i.e. it runs on
CI's primary leg and skips on a stale dev env (both local venvs are <1.9). The `@slow` publisher smoke
lives in `test_publish_ghost_gk_variant.py`.
"""

import numpy as np
import pytest

from scripts.train_ghost_gk import high_sweeper_stratum_mae, per_provider_high_sweeper_coverage


def test_high_sweeper_stratum_mae_only_over_gt30_labels():
    preds = np.array([[12.0, 34.0], [40.0, 34.0], [45.0, 34.0]])
    labels = {"gk_x": np.array([10.0, 38.0, 44.0]), "gk_y": np.array([34.0, 34.0, 34.0])}
    # Only the 38 and 44 rows count; euclid errors |40-38|=2 and |45-44|=1 -> mean 1.5.
    assert abs(high_sweeper_stratum_mae(preds, labels, threshold=30.0) - 1.5) < 1e-9


def test_high_sweeper_stratum_mae_is_nan_when_no_high_labels():
    preds = np.array([[12.0, 34.0]])
    labels = {"gk_x": np.array([10.0]), "gk_y": np.array([34.0])}
    assert np.isnan(high_sweeper_stratum_mae(preds, labels, threshold=30.0))


def test_per_provider_coverage_counts_gt30_fraction():
    labels = {"gk_x": np.array([10.0, 35.0, 40.0, 12.0])}
    provs = np.array(["sportec", "sportec", "gradientsports", "skillcorner"])
    cov = per_provider_high_sweeper_coverage(labels, provs, threshold=30.0)
    assert cov["sportec"] == 0.5
    assert cov["gradientsports"] == 1.0
    assert cov["skillcorner"] == 0.0


# --- CLI wiring (component-level, matching the trainer's existing test pattern) -------------------
def test_grid_spec_from_args_default_and_extended():
    from scripts.train_ghost_gk import grid_spec_from_args
    from silly_kicks.tracking._ghost_gk import DEFAULT_GHOST_GRID

    assert grid_spec_from_args(None) == DEFAULT_GHOST_GRID
    ext = grid_spec_from_args(52.5)
    assert ext.x_max == 52.5 and ext.nx == 105
    # only x_max is swept; y + resolution stay at the default
    assert (ext.y_min, ext.y_max, ext.resolution) == (
        DEFAULT_GHOST_GRID.y_min,
        DEFAULT_GHOST_GRID.y_max,
        DEFAULT_GHOST_GRID.resolution,
    )


def test_grid_x_max_keys_the_shard_generation():
    # BEHAVIORAL (not a source grep): the 4.77.1 stale-shard trap. A lifted ceiling changes the shard
    # label ROWS, so grid_x_max MUST resolve to a DIFFERENT shard generation, or a re-run silently
    # reuses the stale <=30 m shards. Also assert grid_x_max is actually IN the declared inputs.
    import pathlib
    import tempfile

    from scripts._driver import generation_dir
    from scripts.train_ghost_gk import build_extraction_inputs

    common = {"feature_set": "faithful", "subsample_fps": 1.0, "carrier_params": {"beta": 0.0}, "with_actions": False}
    default_inputs = build_extraction_inputs(grid_x_max=30.0, **common)
    sweeper_inputs = build_extraction_inputs(grid_x_max=52.5, **common)
    assert default_inputs["grid_x_max"] == 30.0 and sweeper_inputs["grid_x_max"] == 52.5
    root = pathlib.Path(tempfile.mkdtemp())
    gen_default = generation_dir(root, token_inputs=default_inputs)
    gen_sweeper = generation_dir(root, token_inputs=sweeper_inputs)
    assert gen_default.name != gen_sweeper.name, "a lifted ceiling must NOT reuse the default's shard generation"


def _parse(*argv: str):
    """The uniform stage CLI's parser (ADR-128) over the ghost pipeline's declared parameters."""
    from scripts._stages import build_parser
    from scripts.train_ghost_gk import PIPELINE

    return build_parser(PIPELINE).parse_args(["--dataset", "x", *argv, "--plan", "--store", "s"])


def test_grid_x_max_cli_arg_exists():
    assert _parse("--grid-x-max", "52.5").grid_x_max == 52.5


def test_variant_arg_accepts_sweeper_names():
    # TF-60 PR3: the trainer must be able to NAME the sweeper variant it produces (plan B1's
    # `--variant sweeper`). The bundled artifact self-describes via grid_spec+feature_set, but the
    # metrics.json diagnostics label reads this, so the name has to be an accepted choice.
    for variant in ("sweeper", "sweeper_position_only"):
        assert _parse("--variant", variant).variant == variant


# --- @slow end-to-end trainer stage-CLI smoke (skips on sklearn<1.9; runs on CI's primary leg) ----
def _write_tiny_trainer_corpus(data_dir):
    """Write a minimal 2-game STAGED flat dataset IN ``data_dir`` (real sportec_slim frames, split +
    velocity-derived; ``shards/<gen>/sportec__<game>.parquet``, ``_actions/``, ``home_teams.json``) the
    ghost trainer can extract + CV over. Returns the {game_id: home_team_id} mapping.

    Verified well-formed by ``test_tiny_trainer_corpus_is_well_formed`` (runnable on any sklearn), so
    only the fit-dependent assertions of the @slow smoke depend on the CI primary leg.
    """
    import json

    import pandas as pd

    from scripts._toy_corpora import TOY_GENERATION
    from silly_kicks.tracking import derive_velocities, resolve_defended_goals
    from silly_kicks.tracking.preprocess import smooth_frames

    df = pd.read_parquet("tests/datasets/tracking/action_context_slim/sportec_slim.parquet")
    df = df[df["__kind"] == "frame"].copy()
    for c in ("is_ball", "is_goalkeeper"):
        df[c] = df[c].astype("boolean").fillna(False)
    home = next(t for (g, p, t), e in resolve_defended_goals(df).resolved.items() if str(p) == "1" and float(e) == 0.0)

    home_teams = {}
    frame_ids = sorted(df["frame_id"].dropna().unique())
    mid = frame_ids[len(frame_ids) // 2]
    for name, mask in [("game_a", df["frame_id"] < mid), ("game_b", df["frame_id"] >= mid)]:
        part = df[mask].copy()
        part["game_id"] = name
        part["source_provider"] = "sportec"
        part = derive_velocities(smooth_frames(part))
        shards = data_dir / "shards" / TOY_GENERATION
        shards.mkdir(parents=True, exist_ok=True)
        part.to_parquet(shards / f"sportec__{name}.parquet")
        home_teams[name] = str(home)
    (data_dir / "_actions").mkdir(parents=True, exist_ok=True)
    (data_dir / "home_teams.json").write_text(json.dumps(home_teams), encoding="utf-8")
    return home_teams


def test_tiny_trainer_corpus_is_well_formed(tmp_path):
    # Verify the @slow smoke's fixture locally (runnable on any sklearn) so only the fit is CI-gated.
    import json

    import pandas as pd

    from scripts._toy_corpora import TOY_GENERATION

    data_dir = tmp_path / "corpus"
    home_teams = _write_tiny_trainer_corpus(data_dir)
    assert set(home_teams) == {"game_a", "game_b"}
    assert json.loads((data_dir / "home_teams.json").read_text(encoding="utf-8")) == home_teams
    for name in home_teams:
        f = pd.read_parquet(data_dir / "shards" / TOY_GENERATION / f"sportec__{name}.parquet")
        assert {"vx", "vy", "is_goalkeeper", "source_provider"} <= set(f.columns)
        assert bool(f["is_goalkeeper"].astype("boolean").fillna(False).any())


@pytest.mark.slow
@pytest.mark.skipif(
    not __import__("scripts._train_guard", fromlist=["sklearn_supports_training"]).sklearn_supports_training(),
    reason="the ghost trainer fits real models; needs sklearn in the supported fit range [1.9, 2)",
)
def test_trainer_main_smoke_threads_grid_and_metrics(tmp_path):
    """End-to-end: run the trainer's stage CLI with --grid-x-max 52.5 on the tiny corpus and assert the
    grid reaches the SAVED metadata + the metrics blocks are present. This is the behavioral end-to-end
    proof of the CLI->grid_spec->prepare/model->metadata/metrics glue (TF-60 PR3). Skipped on sklearn<1.9."""
    import json

    from tests.scripts._parity import cli

    _write_tiny_trainer_corpus(tmp_path / "corpus")
    out = tmp_path / "out"

    args = [
        "--dataset",
        "corpus",
        "--grid-x-max",
        "52.5",
        "--cv-folds",
        "2",
        "--n-estimators",
        "20",
        "--max-depth",
        "3",
        "--subsample-fps",
        "25",  # keep dense samples for CV/parity
        "--skip-permutation-importance",
    ]
    cli("scripts.train_ghost_gk", tmp_path / "store", tmp_path, args, "--run-all", "--out", str(out))

    meta = json.loads((out / "metadata.json").read_text())
    assert meta["grid_spec"]["x_max"] == 52.5  # the CLI ceiling reached the SAVED model
    metrics = json.loads((out / "metrics.json").read_text())
    # ADR-067 M4 caveat emitted by the trainer (combined-cycle-completion spec 0.11): a staged corpus
    # carries no public-visibility proof, so it is restricted, and the note names its own commit.
    assert metrics["reproducibility"] == "restricted"
    assert metrics["training_commit"] in metrics["reproducibility_note"]
    assert metrics["grid_x_max"] == 52.5
    assert "high_sweeper_stratum_mae_mean" in metrics
    assert "per_provider_high_sweeper_coverage" in metrics


def test_the_cache_combine_stores_a_mixed_provider_corpus_with_the_legacy_groups(tmp_path):
    """The legacy grouped on `np.array(combined["_game_id"].tolist())`: on a mixed corpus numpy makes every
    id a string. The combined cache stores exactly those strings, so the folds do not move."""
    import numpy as np
    import pandas as pd

    from scripts._stages import LocalStore, read_result, write_result
    from scripts.train_ghost_gk import _combine_packs

    a = pd.DataFrame({"_game_id": [10502, 10502], "f": [1.0, 2.0]})
    b = pd.DataFrame({"_game_id": ["J03WMX"], "f": [3.0]})
    legacy = np.array(pd.concat([a, b], ignore_index=True)["_game_id"].tolist())
    out = _combine_packs([a, b])
    store = LocalStore(tmp_path)
    write_result(store, "s/_output", out)
    back = read_result(store, "s/_output")
    assert np.array(back["_game_id"].tolist()).tolist() == legacy.tolist()
    assert _combine_packs([a])["_game_id"].tolist() == [10502, 10502]
