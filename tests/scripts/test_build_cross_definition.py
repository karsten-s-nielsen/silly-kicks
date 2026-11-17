"""The pre-declared rules of `build_cross_definition` (ADR-133): the start line and the ratio bar."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts import build_cross_definition as m
from silly_kicks.spadl import config as spadlconfig

_ID = spadlconfig.actiontype_id


def test_f1():
    rule = np.array([True, True, False, False])
    native = np.array([True, False, True, False])
    assert m.f1(rule, native) == pytest.approx(0.5)
    assert m.f1(np.zeros(3, bool), np.zeros(3, bool)) == 0.0


def test_selection_rule():
    assert m.CANDIDATE_LINES == (52.5, 60.0, 65.0, 70.0, 75.0, 80.0)
    lines = m.CANDIDATE_LINES
    sb = dict(zip(lines, [0.60, 0.70, 0.66, 0.50, 0.45, 0.40], strict=True))
    gs = dict(zip(lines, [0.40, 0.50, 0.52, 0.30, 0.25, 0.20], strict=True))
    assert m.select_start_line({"statsbomb": sb, "gradientsports": gs}) == 60.0  # mean 0.60 beats 0.59
    # IDSSE is reported but not averaged.
    assert m.select_start_line({"statsbomb": sb, "gradientsports": gs, "idsse": dict.fromkeys(lines, 1.0)}) == 60.0


def test_selection_tie_picks_the_larger_line():
    lines = m.CANDIDATE_LINES
    sb = dict(zip(lines, [0.60, 0.700, 0.704, 0.50, 0.45, 0.40], strict=True))
    gs = dict(zip(lines, [0.40, 0.500, 0.500, 0.30, 0.25, 0.20], strict=True))
    assert m.select_start_line({"statsbomb": sb, "gradientsports": gs}) == 65.0  # within 0.005 of the best


def test_ratio_bar():
    ok = m.ratio_bar({"gradientsports": 16.5, "idsse": 19.0, "skillcorner": 21.0})
    assert ok["passes"] and ok["ratio"] == pytest.approx(21 / 16.5)
    bad = m.ratio_bar({"gradientsports": 10.0, "idsse": 19.0, "skillcorner": 21.0})
    assert not bad["passes"] and bad["lowest"] == "gradientsports" and bad["highest"] == "skillcorner"


def test_delivery_rows_drop_synthetic_dribbles_before_retyping():
    # A wide pass, a synthetic carry (no original_event_id), then a shot in the box: the rule must see the shot
    # as the next action, as the converter did before _add_dribbles ran.
    a = pd.DataFrame(
        {
            "game_id": 1,
            "period_id": 1,
            "action_id": [0, 1, 2],
            "time_seconds": [1.0, 2.0, 3.0],
            "team_id": 1,
            "player_id": [10, 11, 11],
            "original_event_id": ["a", None, "b"],
            "type_id": [_ID["pass"], _ID["dribble"], _ID["shot"]],
            "start_x": [90.0, 92.0, 95.0],
            "start_y": [5.0, 6.0, 34.0],
            "end_x": [92.0, 95.0, 105.0],
            "end_y": [6.0, 34.0, 34.0],
            "cross_native": pd.array([False, pd.NA, pd.NA], dtype="boolean"),
        }
    )
    rows = m._delivery_rows(a, provider="gradientsports", match="1", native_label=True)
    assert rows["type_geo"].tolist() == ["cross"]
    assert rows["cross_native"].tolist() == [False]  # the provider's own label, read from the stage


def test_delivery_rows_refuse_a_stage_from_before_adr_133():
    a = pd.DataFrame({"type_id": [_ID["pass"]], "start_x": [90.0], "start_y": [5.0]})
    with pytest.raises(ValueError, match="no cross_native column"):
        m._delivery_rows(a, provider="skillcorner", match="7", native_label=False)


def test_report_shape_on_the_toy_corpus(tmp_path):
    from scripts._toy_corpora import TOY_BUILDERS
    from tests.scripts._parity import run_serial

    data = TOY_BUILDERS["scripts.build_cross_definition"](tmp_path / "data")
    run_serial(
        "scripts.build_cross_definition",
        tmp_path / "s",
        data,
        ["--dataset", "toyxc", "--cross-start-x", "52.5"],
        tmp_path / "o",
    )
    rep = json.loads((tmp_path / "o" / "cross_definition.json").read_text())
    assert set(rep) == {
        "candidate_lines",
        "f1_by_vendor",
        "selected_start_line",
        "reporting_line",
        "median_crosses_per_match",
        "ratio_bar",
        "n_matches",
        "generation",
        "n_attempted",
        "n_failed",
        "n_counters_unrecorded",
        "n_excluded",
        "run_commit",
        "run_tree_dirty",
        "run_tree_state",
    }
    # ADR-037: the committed artifact must trace to the code that ran (test_artifact_provenance_output): the values
    # are this checkout's own, not placeholders.
    from scripts._provenance import git_provenance

    git = git_provenance()
    assert rep["run_commit"] == git["commit"]
    assert rep["run_tree_dirty"] is git["dirty"] and rep["run_tree_state"] == git["tree_state"]
    n_flat = sum(rep["n_matches"].get(p, 0) for p in ("gradientsports", "idsse", "skillcorner"))
    assert (rep["n_attempted"], rep["n_failed"], rep["n_excluded"], rep["n_counters_unrecorded"]) == (n_flat, 0, 0, 0)
    assert rep["generation"]
    assert rep["reporting_line"] == 52.5 and rep["selected_start_line"] is None  # no StatsBomb arm in the toy
    assert rep["median_crosses_per_match"] and rep["ratio_bar"]["bar"] == m.RATIO_BAR
