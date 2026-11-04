"""TF-46 Task 11 (WS-B): toy synthesized frames carry a current ``stoppage_state`` / ``stoppage_source``.

A toy is SkillCorner / Gradient Sports (non-observed, no restart). Without the stoppage columns a reader
gate (:func:`silly_kicks.tracking.has_stoppage_state`) raises "no observed dead-ball state" on such frames.
The toy builders mark the frames live with :func:`tests._stoppage_fixtures.mark_all_live`, so every reader
and ``prepare_*_training_data`` / ``align_events_to_frames`` finds an observed dead-ball state.
"""

from __future__ import annotations

import pandas as pd

from silly_kicks.tracking import STOPPAGE_SOURCES, has_stoppage_state


def _assert_live_stoppage(frames: pd.DataFrame, *, where: str) -> None:
    assert "stoppage_state" in frames.columns, f"{where}: no stoppage_state column"
    assert "stoppage_source" in frames.columns, f"{where}: no stoppage_source column"
    # Does not raise and returns True (the column is present + the rule version is current).
    assert has_stoppage_state(frames, where=where) is True
    assert (frames["stoppage_state"].astype(object) == "live").all()
    assert set(frames["stoppage_source"].dropna().astype(object)) <= set(STOPPAGE_SOURCES)


def _only_shard(root) -> pd.DataFrame:
    import pathlib

    shards = sorted(pathlib.Path(root).glob("shards/**/*.parquet"))
    assert shards, f"no shard parquet under {root}"
    return pd.read_parquet(shards[0])


def test_game_frames_carry_stoppage_state_and_source():
    from scripts._toy_corpora import _game_frames

    frames = _game_frames("100", "skillcorner", n_frames=3, seed=0)
    _assert_live_stoppage(frames, where="toy._game_frames")


def test_xshot_corpus_frames_carry_stoppage(tmp_path):
    from scripts._toy_corpora import build_xshot_corpus

    build_xshot_corpus(tmp_path, providers=("skillcorner",), episodes=2, heavy_episodes=None)
    _assert_live_stoppage(_only_shard(tmp_path), where="toy.build_xshot_corpus")


def test_xcross_corpus_frames_carry_stoppage(tmp_path):
    from scripts._toy_corpora import build_xcross_corpus

    build_xcross_corpus(tmp_path, providers=("gradientsports",), episodes=2, heavy_episodes=None)
    _assert_live_stoppage(_only_shard(tmp_path), where="toy.build_xcross_corpus")
