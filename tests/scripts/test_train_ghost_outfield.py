"""TF-60 PR5 --- ghost-outfield trainer + publisher tests.

Component-level (non-slow): CLI args, feature_set threading, the stale-shard-token proof
(feature_set keys the shard generation), and a tiny-corpus-well-formed check (runnable on any
sklearn). PLUS one @slow end-to-end ``main()`` smoke that fits a real model and is therefore SKIPPED
where sklearn is outside the supported fit range [1.9, 2) -- it runs on CI's primary leg and skips on
a stale dev env (both local venvs are < 1.9).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts._train_guard import sklearn_supports_training
from scripts.publish_ghost_outfield import main as publish_main
from scripts.train_ghost_outfield import (
    _actions_for,
    _home_team_for,
    _subsample_frames,
    extract_match,
    extraction_inputs,
    feature_set_for_variant,
    main,
    parse_args,
)

# --------------------------------------------------------------------------- #
# Tiny corpus fixture (2 games x jittered 2-team frames with team_in_possession)
# --------------------------------------------------------------------------- #


def _base_players():
    def r(pid, team, x, y, *, gk=False, ball=False, vx=0.2, vy=-0.1):
        return {
            "period_id": 1,
            "team_id": (pd.NA if ball else team),
            "player_id": (pd.NA if ball else pid),
            "is_ball": ball,
            "is_goalkeeper": gk,
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "team_in_possession": 1,
        }

    return [
        r(101, 1, 3.0, 34.0, gk=True),
        r(102, 1, 20.0, 10.0),
        r(103, 1, 22.0, 27.0),
        r(104, 1, 24.0, 44.0),
        r(105, 1, 26.0, 60.0),
        r(106, 1, 45.0, 20.0),
        r(107, 1, 55.0, 40.0),
        r(201, 2, 102.0, 34.0, gk=True),
        r(202, 2, 18.0, 30.0),
        r(203, 2, 30.0, 40.0),
        r(204, 2, 55.0, 25.0),
        r(205, 2, 70.0, 50.0),
        r(206, 2, 85.0, 22.0),
        r(pd.NA, pd.NA, 55.0, 30.0, ball=True, vx=1.5, vy=0.5),
    ]


def _tiny_frames(game_id: str, provider: str = "toy", n_frames: int = 25) -> pd.DataFrame:
    base = _base_players()
    rows = []
    for k in range(n_frames):
        dx = 3.0 * np.sin(0.3 * k)
        dy = 2.0 * np.cos(0.4 * k)
        for p in base:
            row = dict(p)
            row["game_id"] = game_id
            row["frame_id"] = 1000 + k
            row["time_seconds"] = 100.0 + 0.5 * k
            row["source_provider"] = provider  # for the StratifiedGroupKFold(match+provider) CV
            if not row["is_ball"]:
                row["x"] = float(min(104.0, max(1.0, row["x"] + dx)))
                row["y"] = float(min(67.0, max(1.0, row["y"] + dy)))
            rows.append(row)
    return pd.DataFrame(rows)


def _write_tiny_corpus(data_dir, n_games: int = 4):
    # Two providers x two games each, so StratifiedGroupKFold + per-provider MAE actually run.
    for g in range(n_games):
        provider = "provA" if g % 2 == 0 else "provB"
        d = data_dir / f"g{g}"
        d.mkdir(parents=True, exist_ok=True)
        _tiny_frames(f"G{g}", provider=provider).to_parquet(d / "frames.parquet")


# --------------------------------------------------------------------------- #
# Component-level (non-slow)
# --------------------------------------------------------------------------- #


def test_variant_arg_choices_and_feature_set_mapping():
    a = parse_args(["--data-dir", "x", "--variant", "position_only"])
    assert a.variant == "position_only"
    assert feature_set_for_variant("position_only") == "position_only"
    assert feature_set_for_variant("default") == "faithful"


def test_extraction_inputs_feature_set_keys_the_shard_generation():
    """Stale-shard rule (4.77.1): a position_only run drops 4 columns, so its shard token MUST differ."""
    assert extraction_inputs("default") != extraction_inputs("position_only")
    assert extraction_inputs("position_only")["feature_set"] == "position_only"
    assert extraction_inputs("default")["feature_set"] == "faithful"


def test_extraction_inputs_subsample_fps_keys_the_shard_generation():
    """Stale-shard rule: a different fps thins the frames to a different ROW SET, so 1 fps and 25 fps
    shards are NOT interchangeable and the token MUST differ."""
    assert extraction_inputs("default", 1.0) != extraction_inputs("default", 25.0)
    assert extraction_inputs("default", 1.0)["subsample_fps"] == 1.0


def test_subsample_frames_thins_by_fps():
    """1 fps keeps every ``round(frame_rate/fps)``-th unique frame per (game, period); ``None`` is a
    no-op. Mirrors the ghost-GK trainer -- 25 fps tracking is far more (near-duplicate) rows than a
    mean-positioning model needs."""
    rows = []
    for fid in range(30):  # 30 frames @ 10 fps, one period -> step 10 at 1 fps -> keep 3
        for team, pid in ((1, 11), (2, 21)):
            rows.append(
                {
                    "game_id": "g",
                    "period_id": 1,
                    "frame_id": 1000 + fid,
                    "team_id": team,
                    "player_id": pid,
                    "frame_rate": 10.0,
                }
            )
    frames = pd.DataFrame(rows)
    kept = _subsample_frames(frames, 1.0)
    assert kept["frame_id"].nunique() == 3  # frames 1000, 1010, 1020
    assert _subsample_frames(frames, None)["frame_id"].nunique() == 30  # None -> keep all


def test_tiny_corpus_is_well_formed(tmp_path):
    """Runnable on any sklearn: proves the @slow smoke's fixture extracts rows (only the fit is CI-gated)."""
    _write_tiny_corpus(tmp_path)
    frames = pd.read_parquet(tmp_path / "g0" / "frames.parquet")
    feats = extract_match(frames, None, "faithful")
    assert feats is not None and len(feats) > 0
    assert set(feats["slot_index"]) == {1.0, 2.0, 3.0, 4.0}


def _tiny_actions(game_id: str, *, scorer_team: int = 1, goal_time: float = 105.0) -> pd.DataFrame:
    """One goal (shot+success) by ``scorer_team`` + a filler pass -> ``score_diff`` is non-zero after
    ``goal_time`` (frames span t=100..112), so a resolved-actions extract yields a LIVE ``score_diff``."""
    return pd.DataFrame(
        [
            {
                "game_id": game_id,
                "period_id": 1,
                "time_seconds": 101.0,
                "team_id": 1,
                "type_name": "pass",
                "result_name": "success",
            },
            {
                "game_id": game_id,
                "period_id": 1,
                "time_seconds": goal_time,
                "team_id": scorer_team,
                "type_name": "shot",
                "result_name": "success",
            },
        ]
    )


def test_actions_and_home_resolve_tc3_cache_layout(tmp_path):
    """Regression for the real ghost_cache layout: ``{provider}/{game}/frames.parquet`` + ``meta.json``
    + a SEPARATE flat ``_actions/{game}.parquet``.

    The toy flat ``{game}.parquet`` layout masked TWO trainer defects on the real corpus:
    (a) ``_actions_for`` matched nothing under the nested layout, so every ``phase`` / ``score_diff``
    trained constant-0; (b) ``home`` came from ``ids.iloc[0]`` (arbitrary first team), flipping
    ``score_diff``'s sign vs the ``meta.json`` home (measured divergent on IDSSE). Both go RED here
    without the fix.
    """
    data = tmp_path / "cache"
    game = "10502"
    gdir = data / "provX" / game
    gdir.mkdir(parents=True)
    _tiny_frames(game, provider="provX").to_parquet(gdir / "frames.parquet")
    (gdir / "meta.json").write_text(json.dumps({"home_team_id": 2}))  # home=2, but first non-ball team=1
    adir = data / "_actions"
    adir.mkdir()
    _tiny_actions(game, scorer_team=1).to_parquet(adir / f"{game}.parquet")

    args = parse_args(["--data-dir", str(data), "--actions-dir", str(adir)])
    fp = gdir / "frames.parquet"
    frames = pd.read_parquet(fp)

    # (a) actions RESOLVE on the nested tc3 layout (was None -> phase/score_diff dead)
    acts = _actions_for(fp, args)
    assert acts is not None and len(acts) > 0

    # (b) home is the meta.json value (2), NOT the first non-ball team_id (1); the NA ball row upcasts
    # team_id to float, so compare canonically (ADR-019 float-upcast trap).
    assert str(_home_team_for(fp, frames)) == "2"
    assert float(frames[~frames["is_ball"].astype(bool)]["team_id"].dropna().iloc[0]) == 1.0

    # score_diff is LIVE once actions resolve (constant-0 was the dead-feature bug) ...
    feats_home1 = extract_match(frames, acts, "faithful", home_team_id=1)
    feats_home2 = extract_match(frames, acts, "faithful", home_team_id=2)
    assert feats_home1 is not None and feats_home2 is not None
    assert feats_home1["score_diff"].nunique() >= 2
    # ... and its sign is home-perspective: team 1 scored, so home=1 sees +, home=2 sees - .
    assert feats_home1["score_diff"].max() > 0
    assert feats_home2["score_diff"].min() < 0


# --------------------------------------------------------------------------- #
# @slow end-to-end main() smoke (skips on sklearn < 1.9; runs on CI's primary leg)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@pytest.mark.skipif(
    not sklearn_supports_training(),
    reason="bundled-weights fit needs scikit-learn in [1.9, 2)",
)
def test_train_ghost_outfield_main_smoke(tmp_path):
    data = tmp_path / "data"
    _write_tiny_corpus(data)
    out = tmp_path / "out"
    main(
        [
            "--data-dir",
            str(data),
            "--output-dir",
            str(out),
            "--variant",
            "default",
            "--n-estimators",
            "20",
            "--max-depth",
            "3",
            "--cv-folds",
            "2",
            "--allow-dirty",
        ]
    )
    from silly_kicks.tracking import GhostOutfieldModel

    GhostOutfieldModel.load(out / "default")  # guards pass
    m = json.loads((out / "default" / "metrics.json").read_text())
    assert "cv_mae" in m and "coherence" in m
    # Per-provider MAE (spec §6): both toy providers appear as keys.
    assert set(m["cv_mae_by_provider"]) == {"provA", "provB"}
    # Possession-conditioned (Option 2): both regimes reported, so the rest-defense (in-possession)
    # MAE is separable from the out-of-possession line.
    assert set(m["cv_mae_by_possession"]) == {"in_possession", "out_of_possession"}
    assert m["run_tree_dirty"] is True  # ran with --allow-dirty; the fact is recorded, never laundered


def test_publish_verify_only_and_contract_refusal(tmp_path):
    # Non-slow: the toy fit works on any sklearn (the shippable-weights guard is in the TRAINER's
    # main(), not in the model's fit()); --verify-only does no network I/O.
    from silly_kicks.tracking._ghost_outfield import GhostOutfieldModel

    art = tmp_path / "art"
    frames = _tiny_frames("G0", n_frames=40)
    GhostOutfieldModel(n_estimators=15, max_depth=3).fit(frames, None, home_team_id=1).save(art)

    # --verify-only: loads + verifies (SHA/chirality/contract), no upload, no network.
    publish_main(["--artifact-dir", str(art), "--verify-only"])

    # Contract-less artifact is REFUSED (the 4.94.0-class re-break of from_hub).
    meta = json.loads((art / "metadata.json").read_text())
    del meta["feature_contract"]
    with open(art / "metadata.json", "w", newline="\n") as f:
        json.dump(meta, f, indent=2)
    with pytest.raises(SystemExit):
        publish_main(["--artifact-dir", str(art), "--verify-only"])
