"""The trainers' stoppage record (scripts._study_shared; ADR-130)."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts._study_shared import STOPPAGE_SIDE_PREFIX, stoppage_block, stoppage_side_columns
from silly_kicks.tracking._stoppages import STOPPAGE_RULE_VERSION, STOPPAGE_SUMMARY_KEYS


def _frames(states):
    n = len(states)
    return pd.DataFrame(
        {
            "game_id": 1,
            "period_id": 1,
            "frame_id": range(n),
            "time_seconds": [0.1 * k for k in range(n)],
            "stoppage_state": states,
            "stoppage_source": ["events"] * n,
        }
    )


def test_the_side_columns_carry_one_match_counts():
    side = stoppage_side_columns(
        _frames(["live", "dead", "dead", "unknown", "live"]),
        horizon_seconds=0.15,
        label_times=pd.DataFrame({"period_id": [1], "time_seconds": [0.12]}),
    )
    assert set(side) == {f"{STOPPAGE_SIDE_PREFIX}{k}" for k in STOPPAGE_SUMMARY_KEYS}
    assert side["_st_frames_dead"] == 2 and side["_st_frames_unknown"] == 1
    assert side["_st_frames_horizon"] == 0  # frame 2 is dead; no live frame reaches frame 3 within 0.15 s
    assert side["_st_labels_in_dropped"] == 1 and side["_st_source_events"] == 5


def test_the_block_sums_each_match_once_per_provider():
    side_a = stoppage_side_columns(
        _frames(["dead", "live"]), horizon_seconds=1.0, label_times=pd.DataFrame({"period_id": [], "time_seconds": []})
    )
    side_b = stoppage_side_columns(
        _frames(["dead", "dead", "live"]),
        horizon_seconds=1.0,
        label_times=pd.DataFrame({"period_id": [], "time_seconds": []}),
    )
    rows = [{"_provider": "skillcorner", "_match_id": "a", **side_a}] * 5  # five training rows of one match
    rows += [{"_provider": "skillcorner", "_match_id": "b", **side_b}] * 2
    rows += [{"_provider": "sportec", "_match_id": "c", **dict.fromkeys(side_a, 0)}]
    block = stoppage_block(pd.DataFrame(rows))
    assert block["rule_version"] == STOPPAGE_RULE_VERSION
    assert block["providers"]["skillcorner"]["frames_dead"] == 3  # 1 + 2, not 5 + 4
    assert block["providers"]["sportec"]["frames_dead"] == 0


def test_no_stoppage_column_gives_zero_counts():
    fr = _frames(["live"]).drop(columns=["stoppage_state", "stoppage_source"])
    side = stoppage_side_columns(
        fr, horizon_seconds=1.0, label_times=pd.DataFrame({"period_id": [1], "time_seconds": [0.0]})
    )
    assert set(side.values()) == {0}


# ---- IMPL-B01: the xShot extract load fails closed (the xCross one is in test_train_xcross_attempt.py) ----


def _xshot_load(tmp_path, provider: str, *, stoppage: bool):
    import json

    from scripts import train_xshot_occurrence as xs
    from scripts._stages import LocalStore, make_context
    from scripts._toy_corpora import TOY_GENERATION
    from tests.scripts._stage_harness import PROV, params_from_args

    root = tmp_path / "data" / "toy"
    (root / "shards" / TOY_GENERATION).mkdir(parents=True)
    (root / "_actions").mkdir()
    frames = pd.DataFrame(
        {
            "game_id": ["g7"] * 2,
            "is_ball": [True, False],
            "team_id": [pd.NA, "HOME"],
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "source_provider": [provider] * 2,
        }
    )
    if stoppage:
        frames = frames.assign(stoppage_state="live", stoppage_source="events")
    frames.to_parquet(root / "shards" / TOY_GENERATION / f"{provider}__g7.parquet")
    pd.DataFrame({"game_id": ["g7"], "type_id": [0]}).to_parquet(root / "_actions" / f"{provider}__g7.parquet")
    (root / "home_teams.json").write_text(json.dumps({"g7": "HOME"}), encoding="utf-8")
    args = ["--dataset", "toy", "--arms-json", '{"arms": {"public": ["x"]}}']
    ctx = make_context(
        xs.PIPELINE,
        params_from_args(xs.PIPELINE, args),
        PROV,
        store=LocalStore(tmp_path / "s"),
        data_root=tmp_path / "data",
    )
    (ref,) = xs._extract_refs(ctx)
    return xs._extract_load(ref, ctx)


@pytest.mark.parametrize("provider", ["skillcorner", "gradientsports"])
def test_the_xshot_load_refuses_an_unobserved_provider_without_stoppage_columns(tmp_path, provider):
    with pytest.raises(ValueError, match="stoppage_state"):
        _xshot_load(tmp_path, provider, stoppage=False)


def test_the_xshot_load_passes_an_observed_provider_and_a_staged_match(tmp_path):
    assert _xshot_load(tmp_path / "a", "sportec", stoppage=False)[0] == "sportec"
    assert _xshot_load(tmp_path / "b", "skillcorner", stoppage=True)[0] == "skillcorner"
