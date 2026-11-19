"""The loader's SkillCorner events build on a REAL match (pining). Two defects of this cycle passed every synthetic
test and broke this path on every real match (``preserve_native=True``; the running-score list against the
inferred defensive rows). This e2e test runs the path the stage runs.

Asserts: the build runs; the native rows carry the provider's running score (``team_score`` /
``opponent_team_score``) and the derived rows carry NaN there; the score lookup at the end of the match equals the
final score of the match file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


def _final_score(meta: dict) -> tuple[int, int] | None:
    home, away = meta.get("home_team_score"), meta.get("away_team_score")
    if home is None or away is None:
        return None
    return int(home), int(away)


def test_skillcorner_events_build_carries_the_running_score_and_ends_at_the_final_score(tmp_path):
    import scripts._loader_pining as lp
    from silly_kicks.tracking._ghost_gk import _build_score_lookup

    try:
        refs = lp.list_match_refs(providers=["skillcorner"], max_per_provider=1)
    except Exception as exc:  # no token / no network: an e2e test of real data cannot run
        pytest.skip(f"pining unavailable: {type(exc).__name__}")
    if not refs:
        pytest.skip("no skillcorner match available from pining")
    ref = refs[0]
    loaded = lp.load_match(ref, events_only=True, cache_dir=tmp_path)
    actions, home = loaded.actions, str(loaded.home_team_id)
    assert len(actions) > 0

    # the native rows carry the running score; a derived row has no event of its own, so NaN
    assert {"team_score", "opponent_team_score"} <= set(actions.columns), "the feed's running score is not carried"
    native = actions[actions["action_provenance"] == "native"]
    derived = actions[actions["action_provenance"] == "derived"]
    assert len(derived) > 0  # non-vacuity: a real match has inferred defensive and restart rows
    assert native["team_score"].notna().all() and native["opponent_team_score"].notna().all()
    assert derived["team_score"].isna().all() and derived["opponent_team_score"].isna().all()

    # the score lookup at the end of the match equals the match file's final score
    meta_file = next(p for p in Path(tmp_path).rglob("*") if p.name.endswith(("match.json", "_metadata.json")))
    final = _final_score(json.loads(meta_file.read_text(encoding="utf-8")))
    if final is None:
        pytest.skip(f"match {ref.match_id}: the match file has no final score")
    last = actions.sort_values(["period_id", "time_seconds"]).iloc[-1]
    score = _build_score_lookup(actions, home_team_id=home)(
        last["game_id"], int(last["period_id"]), float(last["time_seconds"]) + 600.0
    )
    assert not math.isnan(score) and score == final[0] - final[1], (ref.match_id, score, final)


@pytest.mark.parametrize(("provider", "sources"), [("idsse", {"event"}), ("gradientsports", {"tracking"})])
def test_events_build_runs_on_a_real_match(provider, sources):
    # The events build (the stage's events-only path, spec 4.10) of the other two providers on a real match: it runs,
    # every row has a time source of the provider's set (before the time step), and the key columns are filled.
    import scripts._loader_pining as lp
    from silly_kicks.spadl.schema import TIME_SOURCE_VALUES

    try:
        refs = lp.list_match_refs(providers=[provider], max_per_provider=1)
    except Exception as exc:
        pytest.skip(f"pining unavailable: {type(exc).__name__}")
    if not refs:
        pytest.skip(f"no {provider} match available from pining")
    actions = lp.load_match(refs[0], events_only=True).actions
    assert len(actions) > 100
    assert set(actions["time_source"]) <= set(TIME_SOURCE_VALUES)
    assert set(actions.loc[actions["time_source"] != "derived", "time_source"]) <= sources
    assert actions[["game_id", "period_id", "time_seconds", "type_id"]].notna().all().all()
