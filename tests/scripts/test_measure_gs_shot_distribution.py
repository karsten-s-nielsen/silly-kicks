"""Unit-test the 2a driver's PURE functions without the owner-tier corpus (ADR-054).

`measure_gs_shot_distribution` cannot run in CI: Gradient Sports is owner-tier and the driver
refuses a dirty tree. Its per-match work function and its summariser are pure (DataFrame in,
DataFrame/dict out), so they are testable here -- and testing them is what stops an owner-tier run
being spent discovering a typo.
"""

from __future__ import annotations

import pandas as pd

import silly_kicks.spadl.config as spadlcfg
from scripts.measure_gs_shot_distribution import input_contract, measure_match, summarise

_SHOT = spadlcfg.actiontype_id["shot"]
_PASS = spadlcfg.actiontype_id["pass"]


def _actions(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["team_id", "period_id", "type_id"])


def _item(actions):
    return ("gradientsports", 10517, actions, pd.DataFrame(), 100)


def test_counts_shots_per_team_and_period():
    out = measure_match(_item(_actions([(100, 1, _SHOT)] * 3 + [(200, 1, _SHOT)] * 2 + [(100, 2, _PASS)])))
    assert set(out["n_shots"]) == {3, 2}
    assert list(out["provider"].unique()) == ["gradientsports"]
    assert out["n_shots"].sum() == 5, "non-shot rows must not be counted"


def test_team_ids_are_dense_ranked_so_no_identifier_travels():
    """GS is owner-tier: the driver's contract is that COUNTS travel and identifiers do not."""
    out = measure_match(_item(_actions([(778812, 1, _SHOT), (990001, 1, _SHOT)])))
    assert sorted(out["team_rank"]) == [0, 1]
    assert "team_id" not in out.columns
    flat = out.astype(str).to_numpy().ravel().tolist()
    assert not any("778812" in v or "990001" in v for v in flat), "a real team id leaked into the output"


def test_reliability_flags_use_the_detector_thresholds():
    out = measure_match(_item(_actions([(100, 1, _SHOT)] * 10 + [(200, 1, _SHOT)] * 5 + [(300, 1, _SHOT)] * 4)))
    by_n = dict(zip(out["n_shots"], zip(out["reliable_high"], out["reliable_medium"], strict=True), strict=True))
    assert by_n[10] == (True, True)
    assert by_n[5] == (False, True)
    assert by_n[4] == (False, False)


def test_a_shotless_match_returns_an_EMPTY_frame_not_a_crash():
    """ADR-052: an empty result still writes a shard. Absent means 'not yet run'; present-and-empty
    means 'ran, produced nothing'. Conflating them recomputes every barren match forever."""
    out = measure_match(_item(_actions([(100, 1, _PASS)])))
    assert out.empty
    assert list(out.columns) == [
        "provider",
        "match_id",
        "team_rank",
        "period_id",
        "n_shots",
        "reliable_high",
        "reliable_medium",
    ]


def test_summarise_reports_the_number_the_fixture_defect_turns_on():
    """The committed fixture defers because it has fewer than TWO reliable groups. That count is
    what 2a exists to measure, so it must survive into the summary."""
    a = measure_match(_item(_actions([(100, 1, _SHOT)] * 10 + [(200, 1, _SHOT)] * 10)))
    b = measure_match(_item(_actions([(100, 1, _SHOT)] * 10)))
    b = b.assign(match_id="10518")
    s = summarise(pd.concat([a, b], ignore_index=True))
    assert s["n_matches"] == 2
    assert s["matches_with_ge2_reliable_high"] == 1, "only the two-team match clears the bar"
    assert s["thresholds"] == {"high": 10, "medium": 5}


def test_summarise_survives_an_empty_corpus():
    assert summarise(pd.DataFrame(columns=["provider", "match_id", "n_shots"]))["n_matches"] == 0


def test_the_contract_reads_the_detector_defaults_from_the_LIBRARY():
    """Declared by SYMBOL, not by value: if `detect_input_convention`'s defaults move, the digest
    moves without anyone editing this driver. That is the whole mechanism."""
    c = input_contract()
    assert c["driver"] == "measure_gs_shot_distribution"
    assert c["detector"]["min_shots_per_group_high"] == 10
    assert c["detector"]["min_shots_per_group_medium"] == 5
    assert c["digest"]
