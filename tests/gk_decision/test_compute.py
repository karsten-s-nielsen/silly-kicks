import numpy as np
import pandas as pd
import pytest

from silly_kicks.gk_decision import compute_gk_decision_value, summarize_gk_decision
from silly_kicks.gk_decision._columns import GK_DECISION_SAMPLE_COLUMNS


class _FakeOptionSet:
    def __init__(self, df):
        self._df = df

    def option_rows(self):
        return self._df


_COLS = [
    "game_id",
    "period_id",
    "decision_id",
    "keeper_id",
    "team_id",
    "is_chosen",
    "completion",
    "opponents_bypassed",
    "option_set_source",
]


def _rows(records):
    return pd.DataFrame(records, columns=_COLS)


def test_decision_metrics_known_values():
    # one decision, 3 options; EV = completion*(1+max(0,byp)):
    #   chosen: 1.0*(1+0)=1.0 ; alts: 1.0*(1+1)=2.0 , 1.0*(1+3)=4.0  -> ev_all=[1,2,4]
    recs = [
        ("g", 1, "d1", 9, 7, True, 1.0, 0.0, "native"),
        ("g", 1, "d1", 9, 7, False, 1.0, 1.0, "native"),
        ("g", 1, "d1", 9, 7, False, 1.0, 3.0, "native"),
    ]
    samples, report = compute_gk_decision_value(_FakeOptionSet(_rows(recs)))
    assert list(samples.columns) == list(GK_DECISION_SAMPLE_COLUMNS)
    row = samples.iloc[0]
    assert row["chosen_ev"] == 1.0 and row["best_ev"] == 4.0
    assert row["decision_value"] == pytest.approx(1.0 - (1 + 2 + 4) / 3)  # chosen - mean
    assert row["sel_efficiency"] == pytest.approx(1.0 / 4.0)
    assert row["decision_pct"] == pytest.approx(0.0)  # beats 0 of 2 alternatives
    assert report.n_decisions_in == 1 and report.n_scored == 1


def test_decision_pct_ties_split_half():
    # chosen ties one alternative, beats the other -> (1 + 0.5)/2 = 0.75
    recs = [
        ("g", 1, "d1", 9, 7, True, 1.0, 1.0, "native"),  # EV 2.0
        ("g", 1, "d1", 9, 7, False, 1.0, 1.0, "native"),  # EV 2.0 (tie)
        ("g", 1, "d1", 9, 7, False, 1.0, 0.0, "native"),  # EV 1.0 (beaten)
    ]
    samples, _ = compute_gk_decision_value(_FakeOptionSet(_rows(recs)))
    assert samples.iloc[0]["decision_pct"] == pytest.approx(0.75)


def test_drops_conserve():
    recs = [
        # d1: too few options (2)
        ("g", 1, "d1", 9, 7, True, 1.0, 0.0, "native"),
        ("g", 1, "d1", 9, 7, False, 1.0, 1.0, "native"),
        # d2: no unique chosen (0 chosen)
        ("g", 1, "d2", 9, 7, False, 1.0, 0.0, "native"),
        ("g", 1, "d2", 9, 7, False, 1.0, 1.0, "native"),
        ("g", 1, "d2", 9, 7, False, 1.0, 2.0, "native"),
        # d3: scored
        ("g", 1, "d3", 9, 7, True, 1.0, 0.0, "native"),
        ("g", 1, "d3", 9, 7, False, 1.0, 1.0, "native"),
        ("g", 1, "d3", 9, 7, False, 1.0, 2.0, "native"),
    ]
    _samples, report = compute_gk_decision_value(_FakeOptionSet(_rows(recs)))
    assert report.n_decisions_in == 3
    assert report.n_scored == 1 and report.n_too_few_options == 1 and report.n_no_unique_chosen == 1
    assert report.n_chosen_unvalued == 0
    assert (
        report.n_scored + report.n_too_few_options + report.n_no_unique_chosen + report.n_chosen_unvalued
        == report.n_decisions_in
    )


def test_nan_options_counted_and_explicit():
    # PLAN-02: n_in counts EVERY decision; NaN handling is explicit + conserving.
    recs = [
        # dA: unique chosen but its EV is NaN (NaN completion) -> chosen_unvalued (counted, not vanished)
        ("g", 1, "dA", 9, 7, True, np.nan, 0.0, "native"),
        ("g", 1, "dA", 9, 7, False, 1.0, 1.0, "native"),
        ("g", 1, "dA", 9, 7, False, 1.0, 2.0, "native"),
        # dB: chosen valid, 2 NaN alternatives -> valid subset = 1 < min_options -> too_few (counted)
        ("g", 1, "dB", 9, 7, True, 1.0, 0.0, "native"),
        ("g", 1, "dB", 9, 7, False, np.nan, 1.0, "native"),
        ("g", 1, "dB", 9, 7, False, 1.0, np.nan, "native"),
        # dC: chosen + 2 valid alts + 1 NaN alt -> scored on the finite subset, n_options == 3
        ("g", 1, "dC", 9, 7, True, 1.0, 0.0, "native"),
        ("g", 1, "dC", 9, 7, False, 1.0, 1.0, "native"),
        ("g", 1, "dC", 9, 7, False, 1.0, 2.0, "native"),
        ("g", 1, "dC", 9, 7, False, np.nan, 3.0, "native"),
    ]
    samples, report = compute_gk_decision_value(_FakeOptionSet(_rows(recs)))
    assert report.n_decisions_in == 3
    assert report.n_chosen_unvalued == 1 and report.n_too_few_options == 1 and report.n_scored == 1
    assert (
        report.n_scored + report.n_too_few_options + report.n_no_unique_chosen + report.n_chosen_unvalued
        == report.n_decisions_in
    )
    assert samples.iloc[0]["decision_id"] == "dC" and samples.iloc[0]["n_options"] == 3


def test_keeper_id_dtype_invariance_and_nan_safety():
    # PLAN-01 (ADR-019): the SAME keeper as int in one decision and str in another must canonicalise
    # to ONE keeper (never split the net-of-team / transfer legs); a NaN keeper id must not crash.
    recs = [
        ("g", 1, "d1", 9, 7, True, 1.0, 0.0, "native"),
        ("g", 1, "d1", 9, 7, False, 1.0, 1.0, "native"),
        ("g", 1, "d1", 9, 7, False, 1.0, 2.0, "native"),
        ("g", 2, "d2", "9", 7, True, 1.0, 0.0, "native"),
        ("g", 2, "d2", "9", 7, False, 1.0, 1.0, "native"),
        ("g", 2, "d2", "9", 7, False, 1.0, 2.0, "native"),
        ("g", 3, "d3", np.nan, 7, True, 1.0, 0.0, "native"),
        ("g", 3, "d3", np.nan, 7, False, 1.0, 1.0, "native"),
        ("g", 3, "d3", np.nan, 7, False, 1.0, 2.0, "native"),
    ]
    samples, report = compute_gk_decision_value(_FakeOptionSet(_rows(recs)))
    assert report.n_scored == 3  # NaN-keeper decision still scored (no crash)
    canon = samples[samples["decision_id"].isin(["d1", "d2"])]["keeper"]
    assert canon.nunique() == 1  # int 9 and str "9" collapse to ONE canonical keeper
    agg = summarize_gk_decision(samples)
    assert agg[agg["keeper"] == canon.iloc[0]]["n_decisions"].iloc[0] == 2  # not split across matches


def test_summarize_per_keeper_match():
    recs = [
        ("g", 1, "d3", 9, 7, True, 1.0, 0.0, "native"),
        ("g", 1, "d3", 9, 7, False, 1.0, 1.0, "native"),
        ("g", 1, "d3", 9, 7, False, 1.0, 2.0, "native"),
    ]
    samples, _ = compute_gk_decision_value(_FakeOptionSet(_rows(recs)))
    agg = summarize_gk_decision(samples)
    assert {
        "keeper",
        "game_id",
        "n_decisions",
        "decision_value_mean",
        "sel_efficiency_mean",
        "decision_pct_mean",
    }.issubset(agg.columns)
    assert agg.iloc[0]["n_decisions"] == 1
