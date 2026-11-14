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
import pytest

from scripts import train_xcross_attempt as xc
from scripts._toy_corpora import TOY_GENERATION


def _write_game(
    root: Path,
    *,
    home_map: dict | None,
    game_id: str = "g7",
    stoppage: bool = True,
    provider: str = "skillcorner",
    staged_attempt: bool = True,
) -> None:
    shards = root / "shards" / TOY_GENERATION
    shards.mkdir(parents=True, exist_ok=True)
    (root / "_actions").mkdir(parents=True, exist_ok=True)
    # first non-ball team_id is AWAY -> the first-team fallback would pick AWAY
    frames = pd.DataFrame(
        {
            "game_id": [game_id, game_id, game_id],
            "period_id": [1, 1, 1],
            "frame_id": [1, 1, 1],
            "time_seconds": [0.0, 0.0, 0.0],
            "is_ball": [True, False, False],
            "team_id": [pd.NA, "AWAY", "HOME"],
            "source_provider": [provider] * 3,
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 3.0, 4.0],
        }
    )
    if stoppage:  # ADR-130: a dataset staged with the stoppage columns (tc3-frames-4)
        frames = frames.assign(stoppage_state="live", stoppage_source="events")
    frames.to_parquet(shards / f"{provider}__{game_id}.parquet")
    actions = pd.DataFrame({"game_id": [game_id], "type_name": ["cross"], "type_id": [1]})  # 1 == cross
    if staged_attempt:  # ADR-133: a corpus staged with the tracking cross attempt
        actions["cross_attempt"] = pd.array([pd.NA], dtype="boolean")
    actions.to_parquet(root / "_actions" / f"{provider}__{game_id}.parquet")
    if home_map is not None:
        (root / "home_teams.json").write_text(json.dumps(home_map), encoding="utf-8")


def _loaded(tmp_path: Path, home_map: dict | None, **game):
    from scripts._stages import LocalStore, make_context
    from tests.scripts._stage_harness import PROV, params_from_args

    _write_game(tmp_path / "data" / "toy", home_map=home_map, **game)
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


@pytest.mark.parametrize("provider", ["skillcorner", "gradientsports"])
def test_a_match_without_stoppage_columns_is_refused_for_an_unobserved_provider(tmp_path, provider):
    # ADR-130: the load serves the extract and the probe stages, so both refuse such a dataset.
    with pytest.raises(ValueError, match="stoppage_state"):
        _loaded(tmp_path, None, stoppage=False, provider=provider)


def test_a_match_of_an_observed_provider_needs_no_stoppage_columns(tmp_path):
    prov, *_ = _loaded(tmp_path, None, stoppage=False, provider="sportec")
    assert prov == "sportec"


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


def _label_scene() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three open-play rows with a staged tracking verdict: a SPADL ``pass`` that IS an attempt (a blocked cross),
    a SPADL ``cross`` that is NOT (the ball never reached the box), and one the frames could not judge."""
    from silly_kicks.spadl import config as spc

    actions = pd.DataFrame(
        {
            "game_id": "g",
            "period_id": 1,
            "team_id": "A",
            "time_seconds": [1.0, 5.0, 9.0],
            "type_id": [spc.actiontype_id["pass"], spc.actiontype_id["cross"], spc.actiontype_id["cross"]],
            "cross_attempt": pd.array([True, False, pd.NA], dtype="boolean"),
        }
    )
    frames = pd.DataFrame({"period_id": [1, 1], "frame_id": [1, 2], "time_seconds": [0.0, 10.0]})
    return actions, frames


def test_integrity_counters_count_the_label_v2_events():
    # ADR-126 integrity band: n_events counts the label's own events. Under label v2 (ADR-133) those are the
    # tracking cross attempts, not the SPADL cross type.
    actions, frames = _label_scene()
    assert xc._extract_counters(("skillcorner", "g", actions, frames, "A"), None)["n_events"] == 1


def test_stoppage_label_times_are_the_label_v2_events():
    actions, _frames = _label_scene()
    assert xc._label_times(actions).to_dict("list") == {"period_id": [1], "time_seconds": [1.0]}


def test_the_loader_enriches_once_per_match(tmp_path, monkeypatch):
    # C8: on a corpus staged WITHOUT cross_attempt, the loader computes it once, and the work unit and the
    # integrity counters both read that item instead of computing it again.
    import silly_kicks.tracking._cross_delivery as cd

    calls = []

    def fake(actions, frames, **_kw):
        calls.append(1)
        return actions.assign(cross_attempt=pd.array([pd.NA] * len(actions), dtype="boolean"))

    monkeypatch.setattr(cd, "add_cross_deliveries", fake)
    item = _loaded(tmp_path, None, staged_attempt=False)
    assert calls == [1] and "cross_attempt" in item[2].columns
    xc._extract_counters(item, None)
    assert calls == [1]
