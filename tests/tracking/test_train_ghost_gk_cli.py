"""Trainer CLI wires one carrier cp into prepare + fit (PR-S81 / N1 / NP3).

The recorded==used invariant is unit-tested at the model level (test_ghost_gk_r3)
and the prepare level; this guards the place it is actually WIRED -- the trainer
passing one cp to both prepare and fit -- end to end through the CLI.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS
from tests.tracking.test_ghost_gk import _make_ghost_gk_frames


def _tiny_cache(root: Path, n_games: int = 2) -> None:
    for g in range(n_games):
        gid = f"{100 + g}"
        gdir = root / "test" / gid
        gdir.mkdir(parents=True, exist_ok=True)
        frames = pd.concat(
            [_make_ghost_gk_frames(game_id=gid, frame_id=f, timestamp=float(f)) for f in range(1, 40)],
            ignore_index=True,
        )
        frames.to_parquet(gdir / "frames.parquet")
        (gdir / "meta.json").write_text(json.dumps({"home_team_id": 1}))


def _invoke(data: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
    """One CLI run against an EXPLICIT data/out pair, so a test can invoke it more than once."""
    return subprocess.run(  # noqa: S603
        [
            sys.executable,
            "scripts/train_ghost_gk.py",
            "--data-dir",
            str(data),
            "--output-dir",
            str(out),
            "--n-estimators",
            "10",
            "--cv-folds",
            "2",
            # The trainer refuses a dirty tree (it stamps training_commit into a shipped
            # artifact); a test run is by definition a dev run.
            "--allow-dirty",
            # Metrics-only and the slowest step at this scale; nothing asserted here reads it.
            "--skip-permutation-importance",
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def _corpus(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "cache"
    data.mkdir()
    _tiny_cache(data)
    return data, tmp_path / "out"


def _run(tmp_path: Path, *extra: str) -> dict:
    data, out = _corpus(tmp_path)
    proc = _invoke(data, out, *extra)
    assert proc.returncode == 0, proc.stderr
    return json.loads((out / "ghost_gk_v1" / "metadata.json").read_text())


def _ok(proc: subprocess.CompletedProcess) -> str:
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


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


# ---------------------------------------------------------------------------
# ADR-052: the per-game extraction is sharded, and the whole-corpus cache token was widened
# ---------------------------------------------------------------------------

_HIT = "Loading cached features"
_SKIP = "skip (shard exists)"


@pytest.mark.slow
def test_a_changed_extraction_PARAMETER_misses_the_feature_cache(tmp_path):
    """Both directions, because only the pair is evidence.

    The recorded cache token used to be the penalty-area geometry alone, so a re-run at a different
    ``--subsample-fps`` reused the previous run's feature matrix while ``metadata.json`` recorded
    the NEW parameters -- the recorded==used invariant (PR-S81) broken by the cache beneath it.
    Asserting only the MISS would pass equally well on a token that never hits at all, which would
    replace a silent-staleness bug with a silent-recompute one.
    """
    data, out = _corpus(tmp_path)
    _ok(_invoke(data, out, "--subsample-fps", "25.0"))

    same = _ok(_invoke(data, out, "--subsample-fps", "25.0"))
    assert _HIT in same, "identical parameters must still hit the whole-corpus cache"

    changed = _ok(_invoke(data, out, "--subsample-fps", "12.5"))
    assert _HIT not in changed, "a changed --subsample-fps silently reused the previous features"


@pytest.mark.slow
def test_a_RESUMED_extraction_skips_games_that_already_have_a_shard(tmp_path):
    """Drop the whole-corpus cache token only. The shard loop is entered again and must find every
    game already done -- and produce the same training matrix it produced the first time."""
    data, out = _corpus(tmp_path)
    first = _ok(_invoke(data, out))
    assert _SKIP not in first, "nothing was sharded yet, so nothing could be skipped"
    n_first = json.loads((out / "ghost_gk_v1" / "metrics.json").read_text())["n_samples"]

    cache_dir = out / "ghost_gk_v1" / "_feature_cache"
    (cache_dir / "cache_token.txt").unlink()
    resumed = _ok(_invoke(data, out))

    assert _SKIP in resumed, "the shards were ignored and every game was re-extracted"
    assert json.loads((out / "ghost_gk_v1" / "metrics.json").read_text())["n_samples"] == n_first

    # Non-vacuity: with the shards ALSO gone, the same invocation must do the work again. Without
    # this half, a run that silently produced no shards at all would look like a healthy resume.
    (cache_dir / "cache_token.txt").unlink()
    for shard in cache_dir.rglob("*.parquet"):
        shard.unlink()
    assert _SKIP not in _ok(_invoke(data, out)), "the skip marker does not depend on the shards"


@pytest.mark.slow
def test_a_game_with_NO_home_mapping_is_skipped_and_counted(tmp_path):
    """The item generator's other exit. A game absent from the home-team map is never yielded, so
    it is not an item and correctly leaves no shard -- but it must still be REPORTED, and the
    counter it increments lives in the enclosing scope through a `nonlocal`, which is exactly the
    kind of binding that fails only at runtime.
    """
    data = tmp_path / "cache"
    data.mkdir()
    _tiny_cache(data, n_games=3)
    # Map two of the three games; the third has no home_team_id and must drop out.
    mapped = tmp_path / "home_teams.json"
    mapped.write_text(json.dumps({"100": "1", "101": "1"}), encoding="utf-8")

    out = _ok(_invoke(data, tmp_path / "out", "--home-teams", str(mapped)))

    assert "SKIP game 102: no home_team_id in mapping" in out
    assert "(1 skipped" in out, "the skip was not carried out of the generator into the summary"
    metrics = json.loads((tmp_path / "out" / "ghost_gk_v1" / "metrics.json").read_text())
    assert metrics["n_games"] == 2, "a game with no home mapping still reached the training set"


@pytest.mark.slow
def test_a_CORRECTED_home_team_mapping_invalidates_only_the_games_it_corrects(tmp_path):
    """`home_team_id` drives the goal-relative flip of every feature and label in a shard, so a
    corrected mapping must reach the features.

    It is a PER-ITEM input while `token_inputs` is per-PASS, which is why it lives in the shard KEY
    instead. The failure it closes was silent and total: the shard was skipped, the correction never
    landed, and the whole-corpus cache accepted the result too, so the model trained on wrong-handed
    data with nothing anywhere reporting a problem. Declaring the whole `{game_id: home}` map would
    have "fixed" it by invalidating every shard whenever a match was added to --data-dir -- the
    over-invalidation the selector rule exists to prevent.
    """
    data = tmp_path / "cache"
    data.mkdir()
    _tiny_cache(data, n_games=2)
    out = tmp_path / "out"
    first = tmp_path / "home_a.json"
    first.write_text(json.dumps({"100": "1", "101": "1"}), encoding="utf-8")

    _ok(_invoke(data, out, "--home-teams", str(first)))
    cache_dir = out / "ghost_gk_v1" / "_feature_cache"
    (cache_dir / "cache_token.txt").unlink()  # force the shard loop to be re-entered
    assert _SKIP in _ok(_invoke(data, out, "--home-teams", str(first))), "same mapping must resume"

    # Now CORRECT one game's home team. That game must be re-extracted, not skipped.
    corrected = tmp_path / "home_b.json"
    corrected.write_text(json.dumps({"100": "2", "101": "1"}), encoding="utf-8")
    (cache_dir / "cache_token.txt").unlink()
    out_text = _ok(_invoke(data, out, "--home-teams", str(corrected)))

    assert "game 100" not in out_text or _SKIP not in out_text.split("game 100")[-1].split("\n")[0], (
        "the corrected game was skipped -- the new mapping never reached its features"
    )
    shards = {p.name for p in cache_dir.rglob("*.parquet")}
    assert any("__2" in n for n in shards), f"no shard keyed on the corrected home team: {sorted(shards)}"
    assert any("__1" in n for n in shards), "the unchanged game's shard should still be reusable"
