"""TF-60 PR3 Task 7: trainer glue for the sweeper re-fit (>30 m stratum + coverage + grid wiring).

Component-level (non-slow): the metric helpers, the CLI arg, `grid_spec_from_args`, and the
BEHAVIORAL stale-shard-token proof (`grid_x_max` keys the shard generation). PLUS one `@slow`
end-to-end `main()` smoke (`test_trainer_main_smoke_threads_grid_and_metrics`) that fits a real model
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


def test_grid_x_max_cli_arg_exists():
    import sys

    from scripts.train_ghost_gk import parse_args

    argv = ["prog", "--data-dir", ".", "--grid-x-max", "52.5"]
    old = sys.argv
    try:
        sys.argv = argv
        args = parse_args()
    finally:
        sys.argv = old
    assert args.grid_x_max == 52.5


def test_variant_arg_accepts_sweeper_names():
    # TF-60 PR3: the trainer must be able to NAME the sweeper variant it produces (plan B1's
    # `--variant sweeper`). The bundled artifact self-describes via grid_spec+feature_set, but the
    # metrics.json diagnostics label reads this, so the name has to be an accepted choice.
    import sys

    from scripts.train_ghost_gk import parse_args

    for variant in ("sweeper", "sweeper_position_only"):
        argv = ["prog", "--data-dir", ".", "--variant", variant]
        old = sys.argv
        try:
            sys.argv = argv
            args = parse_args()
        finally:
            sys.argv = old
        assert args.variant == variant


# --- @slow end-to-end trainer main() smoke (skips on sklearn<1.9; runs on CI's primary leg) --------
def _write_tiny_trainer_corpus(data_dir):
    """Write a minimal 2-game flat corpus (real sportec_slim frames, split + velocity-derived) the
    ghost trainer can extract + CV over. Returns the {game_id: home_team_id} mapping.

    Verified well-formed by ``test_tiny_trainer_corpus_is_well_formed`` (runnable on any sklearn), so
    only the fit-dependent assertions of the @slow smoke depend on the CI primary leg.
    """
    import pandas as pd

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
        (data_dir / name).mkdir(parents=True, exist_ok=True)
        part.to_parquet(data_dir / name / "frames.parquet")
        home_teams[name] = str(home)
    return home_teams


def test_tiny_trainer_corpus_is_well_formed(tmp_path):
    # Verify the @slow smoke's fixture locally (runnable on any sklearn) so only the fit is CI-gated.
    import json

    import pandas as pd

    data_dir = tmp_path / "corpus"
    home_teams = _write_tiny_trainer_corpus(data_dir)
    assert set(home_teams) == {"game_a", "game_b"}
    for name in home_teams:
        f = pd.read_parquet(data_dir / name / "frames.parquet")
        assert {"vx", "vy", "is_goalkeeper", "source_provider"} <= set(f.columns)
        assert bool(f["is_goalkeeper"].astype("boolean").fillna(False).any())
    json.dumps(home_teams)  # serializable for --home-teams


@pytest.mark.slow
@pytest.mark.skipif(
    not __import__("scripts._train_guard", fromlist=["sklearn_supports_training"]).sklearn_supports_training(),
    reason="the ghost trainer fits real models; needs sklearn in the supported fit range [1.9, 2)",
)
def test_trainer_main_smoke_threads_grid_and_metrics(tmp_path, monkeypatch):
    """End-to-end: run the trainer main() with --grid-x-max 52.5 on the tiny corpus and assert the grid
    reaches the SAVED metadata + the metrics blocks are present. This is the behavioral end-to-end proof
    of the CLI->grid_spec->prepare/model->metadata/metrics glue (TF-60 PR3). Skipped on sklearn<1.9."""
    import json
    import sys

    data_dir = tmp_path / "corpus"
    home_teams = _write_tiny_trainer_corpus(data_dir)
    ht_path = tmp_path / "home_teams.json"
    ht_path.write_text(json.dumps(home_teams))
    out = tmp_path / "out"

    argv = [
        "train_ghost_gk",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(out),
        "--home-teams",
        str(ht_path),
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
        "--allow-dirty",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    from scripts.train_ghost_gk import main

    main()

    meta = json.loads((out / "ghost_gk_v1" / "metadata.json").read_text())
    assert meta["grid_spec"]["x_max"] == 52.5  # the CLI ceiling reached the SAVED model
    metrics = json.loads((out / "ghost_gk_v1" / "metrics.json").read_text())
    assert metrics["grid_x_max"] == 52.5
    assert "high_sweeper_stratum_mae_mean" in metrics
    assert "per_provider_high_sweeper_coverage" in metrics
