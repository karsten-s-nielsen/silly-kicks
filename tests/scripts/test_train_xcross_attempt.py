"""TF-65 C2: the xcross loader's home override.

xcross signs ``score_differential`` from HOME's perspective, so the first-non-ball ``team_id``
heuristic flips the sign on a match whose frames lead with the away team. A staged flat dataset carries
no meta.json, so its ``home_teams.json`` (a ``{game_id: home_team_id}`` map) must override the heuristic.
Falls back to the first team when the map is absent or lacks the game. ADR-128: the loader is the
``extract`` stage's `_load_match`; the held-out probe pass never applies the training map.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import train_xcross_attempt as xc
from scripts._toy_corpora import TOY_GENERATION


def _write_game(root: Path, *, home_map: dict | None, game_id: str = "g7") -> None:
    shards = root / "shards" / TOY_GENERATION
    shards.mkdir(parents=True, exist_ok=True)
    (root / "_actions").mkdir(parents=True, exist_ok=True)
    # first non-ball team_id is AWAY -> the first-team fallback would pick AWAY
    pd.DataFrame(
        {
            "game_id": [game_id, game_id, game_id],
            "is_ball": [True, False, False],
            "team_id": [pd.NA, "AWAY", "HOME"],
            "source_provider": ["skillcorner", "skillcorner", "skillcorner"],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 3.0, 4.0],
        }
    ).to_parquet(shards / f"skillcorner__{game_id}.parquet")
    pd.DataFrame({"game_id": [game_id], "type_name": ["cross"]}).to_parquet(
        root / "_actions" / f"skillcorner__{game_id}.parquet"
    )
    if home_map is not None:
        (root / "home_teams.json").write_text(json.dumps(home_map), encoding="utf-8")


def _loaded(tmp_path: Path, home_map: dict | None):
    from scripts._stages import LocalStore, make_context
    from tests.scripts._stage_harness import PROV, params_from_args

    _write_game(tmp_path / "data" / "toy", home_map=home_map)
    args = ["--dataset", "toy", "--probe-dataset", "probe", "--arms-json", '{"arms": {"public": ["x"]}}']
    ctx = make_context(
        xc.PIPELINE,
        params_from_args(xc.PIPELINE, args),
        PROV,
        store=LocalStore(tmp_path / "s"),
        data_root=tmp_path / "data",
    )
    (ref,) = xc._extract_refs(ctx)
    return xc._load_match(ref, ctx)


def test_home_map_overrides_first_team(tmp_path):
    prov, mid, *_, home = _loaded(tmp_path, {"g7": "HOME"})
    assert prov == "skillcorner"
    assert mid == "skillcorner-g7"  # the repackaged per-game dir name (the legacy `_source_key` match id)
    assert home == "HOME"


def test_falls_back_to_first_team_without_map(tmp_path):
    *_, home = _loaded(tmp_path, None)
    assert home == "AWAY"


def test_falls_back_when_game_absent_from_map(tmp_path):
    *_, home = _loaded(tmp_path, {"other": "X"})
    assert home == "AWAY"


def test_the_probe_dataset_skips_dot_directories(tmp_path):
    """The 4.138.0 crash: an ``hf download`` ``.cache/`` beside the probe game dirs became a "game". The
    probe stage's refs come from `staged_pergame_refs`, which never treats a dot-directory as a unit."""
    from scripts._stages import LocalStore, make_context
    from scripts._toy_corpora import build_xcross_probe_corpus
    from tests.scripts._stage_harness import PROV, params_from_args

    probe = build_xcross_probe_corpus(tmp_path / "data" / "probe", n_games=2, episodes=1)
    (probe / ".cache" / "huggingface").mkdir(parents=True)
    (probe / ".cache" / "frames.parquet").write_bytes(b"not a parquet")
    args = ["--dataset", "toy", "--probe-dataset", "probe", "--arms-json", '{"arms": {"public": ["x"]}}']
    ctx = make_context(
        xc.PIPELINE,
        params_from_args(xc.PIPELINE, args),
        PROV,
        store=LocalStore(tmp_path / "s"),
        data_root=tmp_path / "data",
    )
    refs = xc._probe_refs(ctx)
    assert [r.mid for r in refs] == ["gradientsports-200", "gradientsports-201"]
