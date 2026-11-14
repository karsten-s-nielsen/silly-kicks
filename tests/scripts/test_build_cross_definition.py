"""The pre-declared rules of `build_cross_definition` (ADR-133; spec §7.2-§7.3, plan Tasks 9-10)."""

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
    assert m.CANDIDATE_LINES == (52.5, 60.0, 65.0, 70.0, 75.0, 80.0)  # spec r7 §7.2 owner ruling (2026-11-14)
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


def test_selection_can_pick_a_new_candidate():
    lines = m.CANDIDATE_LINES
    sb = dict(zip(lines, [0.60, 0.61, 0.62, 0.63, 0.66, 0.64], strict=True))
    gs = dict(zip(lines, [0.40, 0.41, 0.42, 0.43, 0.45, 0.44], strict=True))
    assert m.select_start_line({"statsbomb": sb, "gradientsports": gs}) == 75.0


def test_ratio_bar():
    ok = m.ratio_bar({"gradientsports": 16.5, "idsse": 19.0, "skillcorner": 21.0})
    assert ok["passes"] and ok["ratio"] == pytest.approx(21 / 16.5)
    bad = m.ratio_bar({"gradientsports": 10.0, "idsse": 19.0, "skillcorner": 21.0})
    assert not bad["passes"] and bad["lowest"] == "gradientsports" and bad["highest"] == "skillcorner"


def test_weighted_quantile_gives_each_group_equal_weight():
    big = np.full(1000, 1.0)  # one provider with many rows at 1.0
    small = np.array([10.0, 10.0])  # another with two rows at 10.0
    # Unweighted, the 95th percentile is 1.0; with equal weight per group it is 10.0.
    assert np.quantile(np.concatenate([big, small]), 0.95) == 1.0
    assert m.weighted_quantile({"a": big, "b": small}, 0.95) == 10.0
    assert np.isnan(m.weighted_quantile({"a": np.array([np.nan])}, 0.5))


def _calibration_rows() -> pd.DataFrame:
    """Three providers, 20 applicable rows each (spec r7 §7.4).

    * ``release_dist_m``: IDSSE 0.1-2.0 m (the clean-sync reference); GS 1-20 m and SkillCorner 0.5-10 m carry
      alignment tails, so a pooled 95th percentile would differ from the IDSSE one.
    * ``passer_contact_dt`` rises with the row index, so the release-match filter changes the gap sample.
    * GS rows 0-3 are native crosses with an opponent first contact; rows 0-2 are blocked; row 1's first contact
      is a teammate; row 2's first contact comes after ``block_window_s``. Only row 0 passes every block filter.
    """
    rows = []
    for provider, fr in (("gradientsports", 25.0), ("skillcorner", 10.0), ("idsse", 25.0)):
        for i in range(20):
            dist = {"gradientsports": float(i + 1), "skillcorner": 0.5 * (i + 1), "idsse": 0.1 * (i + 1)}[provider]
            rows.append(
                {
                    "provider": provider,
                    "applicable": True,
                    "release_source": "elastic",
                    "release_dist_m": dist,
                    "passer_contact_dt": 0.01 * (i + 1),
                    "first_contact_dt": 0.3 if i != 2 else 1.5,
                    "first_contact_dist_m": 4.0 + i,
                    "first_contact_is_opponent": i != 1,
                    "cross_native": provider == "gradientsports" and i < 4,
                    "cross_blocked": provider == "gradientsports" and i < 3,
                    "coverage": 0.5 if i < 5 else 1.0,
                    "frame_rate": fr,
                }
            )
    df = pd.DataFrame(rows)
    for col in ("first_contact_is_opponent", "cross_native", "cross_blocked"):
        df[col] = df[col].astype("boolean")
    return df


def test_calibrate_release_match_uses_the_idsse_reference_only():
    rows = _calibration_rows()
    cal = m.calibrate(rows, block_window_s=1.0)
    idsse = rows.loc[rows["provider"] == "idsse", "release_dist_m"].to_numpy()
    assert cal["release_match_m"] == pytest.approx(float(np.quantile(idsse, 0.95)))
    pooled = {str(p): g["release_dist_m"].to_numpy() for p, g in rows.groupby("provider")}
    assert cal["release_match_m"] != pytest.approx(m.weighted_quantile(pooled, 0.95))  # not the contaminated pool
    assert cal["n_release_match_samples"] == 20
    # The other providers' tails are reported, not used.
    assert set(cal["release_dist_p95_by_provider"]) == {"gradientsports", "idsse", "skillcorner"}


def test_calibrate_release_gap_uses_rows_that_pass_the_release_match():
    rows = _calibration_rows()
    cal = m.calibrate(rows, block_window_s=1.0)
    ok = rows[rows["release_dist_m"] <= cal["release_match_m"]]
    gaps = {str(p): g["passer_contact_dt"].to_numpy() for p, g in ok.groupby("provider")}
    assert cal["release_gap_s"] == pytest.approx(max(m.weighted_quantile(gaps, 0.95), 1 / 25.0))
    every = {str(p): g["passer_contact_dt"].to_numpy() for p, g in rows.groupby("provider")}
    assert cal["release_gap_s"] != pytest.approx(max(m.weighted_quantile(every, 0.95), 1 / 25.0))


def test_calibrate_block_radius_keeps_contacts_inside_the_block_window():
    rows = _calibration_rows()
    cal = m.calibrate(rows, block_window_s=1.0)
    assert cal["block_radius_m"] == pytest.approx(4.0)  # row 0 only: row 1 is a teammate, row 2 is late
    assert cal["n_block_samples_gs"] == 1
    late = m.calibrate(rows, block_window_s=2.0)  # the window decides: row 2 (6.0 m at 1.5 s) joins
    assert late["n_block_samples_gs"] == 2
    assert late["block_radius_m"] == pytest.approx(float(np.quantile([4.0, 6.0], 0.95)))


def test_calibrate_keeps_the_fixed_rules():
    cal = m.calibrate(_calibration_rows(), block_window_s=1.0)
    assert cal["direction_window_s"] == pytest.approx(2 / 10.0)  # two frames at the LOWEST frame rate
    assert cal["block_window_s"] == 1.0 and cal["path_cap_s"] == 4.0  # fixed, reported only
    assert cal["coverage_below_0.8_share"] == {"gradientsports": 0.25, "idsse": 0.25, "skillcorner": 0.25}
    assert cal["coverage_owner_decision_needed"] is True


_NAN = float("nan")
_OVER = {"idsse": [1.0] * 200, "gradientsports": [1.0] * 7 + [9.0] * 3}  # 30 % of GS rows beyond a 1.0 m match


@pytest.mark.parametrize(
    ("dists", "event_released", "n", "share", "fires"),
    [
        pytest.param({"idsse": [1.0] * 200}, None, 200, {"idsse": 0.0}, False, id="clean"),
        pytest.param({"idsse": [1.0] * 180}, None, 180, {"idsse": 0.0}, True, id="small_idsse_sample"),
        pytest.param(_OVER, None, 200, {"gradientsports": 0.3, "idsse": 0.0}, True, id="share_alone"),
        pytest.param(
            {"idsse": [1.0] * 200, "gradientsports": [1.0] * 9 + [9.0]},
            None,
            200,
            {"gradientsports": 0.1, "idsse": 0.0},
            False,
            id="share_under_bar",
        ),
        # Review A C12 N10: rows released at the event time are not in the sample (counted, GS would be 0.3).
        pytest.param(_OVER, 9.0, 200, {"gradientsports": 0.0, "idsse": 0.0}, False, id="elastic_rows_only"),
        # Review B R7-B-05: a NaN distance is <NA> whatever the match is: in neither the share nor the sample size.
        pytest.param(
            {"idsse": [1.0] * 200 + [_NAN] * 10, "gradientsports": [1.0] * 7 + [_NAN] * 3},
            None,
            200,
            {"gradientsports": 0.0, "idsse": 0.0},
            False,
            id="nan_ignored",
        ),
        pytest.param({"idsse": [1.0] * 195 + [_NAN] * 10}, None, 195, {"idsse": 0.0}, True, id="nan_not_a_sample"),
    ],
)
def test_release_match_owner_trigger(dists, event_released, n, share, fires):
    # Spec r7 §7.4: fewer than 200 finite IDSSE distances, or a provider whose ELASTIC-confident share beyond the
    # release match exceeds 20 %, fires the trigger.
    tmpl = _calibration_rows().iloc[0].to_dict()
    rows = pd.DataFrame(
        [{**tmpl, "provider": p, "release_dist_m": d, "cross_native": False} for p, ds in dists.items() for d in ds]
    )
    if event_released is not None:
        rows.loc[rows["release_dist_m"] == event_released, "release_source"] = "event"
    cal = m.calibrate(rows, block_window_s=1.0)
    assert cal["release_match_m"] == 1.0
    assert cal["n_release_match_samples"] == n
    assert cal["release_match_fail_share"] == pytest.approx(share)
    assert cal["release_match_owner_decision_needed"] is fires


def test_fixed_point_checks_each_calibrated_field_and_the_start_line():
    params = {"release_match_m": 5.0, "release_gap_s": 0.2, "block_radius_m": 8.0, "direction_window_s": 0.2}
    cal, run = dict(params), params | {"cross_start_x": 70.0}
    # The expected keys are spelled out (spec r7 §7.4), not read from m._CALIBRATED, the constant under test.
    assert m.fixed_point(cal, run, selected_line=70.0) == dict.fromkeys([*params, "cross_start_x", "all"], True)
    off = m.fixed_point(cal | {"block_radius_m": 9.5}, run, selected_line=70.0)
    assert off == {**dict.fromkeys(params, True), "block_radius_m": False, "cross_start_x": True, "all": False}
    # Review A C9: a run at another line than the selected one (an omitted --cross-start-x) is not a fixed point.
    line = m.fixed_point(cal, params | {"cross_start_x": 52.5}, selected_line=70.0)
    assert line["cross_start_x"] is False and line["all"] is False
    assert m.fixed_point(cal, run, selected_line=None)["all"] is False  # an unknown selection


def test_calibrate_without_idsse_reports_nan():
    rows = _calibration_rows()
    cal = m.calibrate(rows[rows["provider"] != "idsse"], block_window_s=1.0)
    assert np.isnan(cal["release_match_m"]) and np.isnan(cal["release_gap_s"])


def test_alignment_classes_partition_the_applicable_rows():
    g = pd.DataFrame(
        {
            "attempt": pd.array([True, False, pd.NA, pd.NA, pd.NA, pd.NA], dtype="boolean"),
            "release_source": ["elastic", "event", "event", "elastic", "event", "event"],
            "release_dist_m": [1.0, 2.0, 9.0, np.nan, 1.0, np.nan],
            "start_x": [90.0, 90.0, 90.0, 90.0, 90.0, np.nan],
            "start_y": 5.0,
            "release_match_cut": 3.0,
        }
    )
    cls = m.alignment_class(g).tolist()
    assert cls == [
        "aligned_elastic",
        "unaligned_but_matched",
        "misaligned",
        "no_ball_at_release",
        "coverage_unanswerable",
        "unknown_start",
    ]
    assert set(cls) == set(m.ALIGNMENT_CLASSES)


def test_alignment_is_split_by_provider_and_row_class():
    # Spec 7.3: the shares are per provider AND row class (open play, corner, free kick); the corner share is what
    # qualifies corner_kind_source == "tracking". An open-play row with an unknown start is judged (unknown_start),
    # an out-of-zone open-play row (not applicable, known start) is not.
    flat = pd.DataFrame(
        {
            "provider": ["gs", "gs", "gs", "gs", "sc"],
            "type_geo": ["cross", "corner_crossed", "pass", "pass", "freekick_short"],
            "applicable": [True, True, False, False, True],
            "attempt": pd.array([True, pd.NA, pd.NA, False, True], dtype="boolean"),
            "release_source": ["elastic", "event", "event", "event", "event"],
            "release_dist_m": [1.0, 9.0, np.nan, np.nan, 1.0],
            "start_x": [90.0, 104.0, np.nan, 30.0, 70.0],
            "start_y": [5.0, 1.0, 5.0, 30.0, 10.0],
            "release_match_cut": 3.0,
        }
    )
    t = m.alignment_table(flat)
    assert set(t) == {"gs", "sc"}
    assert set(t["gs"]) == {"open_play", "corner"} and set(t["sc"]) == {"freekick"}
    assert t["gs"]["open_play"]["n"] == 2  # the in-zone cross + the unknown-start pass, not the out-of-zone pass
    assert t["gs"]["open_play"]["aligned_elastic"] == 0.5 and t["gs"]["open_play"]["unknown_start"] == 0.5
    assert t["gs"]["corner"] == {"n": 1, **dict.fromkeys(m.ALIGNMENT_CLASSES, 0.0), "misaligned": 1.0}
    for cells in t.values():
        for cell in cells.values():
            assert sum(cell[c] for c in m.ALIGNMENT_CLASSES) == pytest.approx(1.0)


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
        }
    )
    rows, _geo = m._delivery_rows(a, provider="gradientsports", match="1", native_label=True)
    assert rows["type_geo"].tolist() == ["cross"]


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
        "tracking_params",
        "calibration",
        "fixed_point",
        "evidence",
        "n_matches",
    }
    assert set(rep["fixed_point"]) == {*m._CALIBRATED, "cross_start_x", "all"}
    # No StatsBomb arm, so no selected line: the run line (52.5) is not a fixed point (review A C12 N3).
    assert rep["fixed_point"]["cross_start_x"] is False
    assert rep["tracking_params"]["block_window_s"] == 1.0
    assert rep["reporting_line"] == 52.5 and rep["selected_start_line"] is None  # no StatsBomb arm in the toy
    ev = rep["evidence"]
    assert {"median_crosses_per_match", "ratio_bar", "cross_attempt", "alignment", "corner_kind_source"} <= set(ev)
    assert ev["alignment"], "precondition: the toy has applicable rows"
    for cells in ev["alignment"].values():
        assert set(cells) <= set(m.ROW_CLASSES)
        for shares in cells.values():
            assert sum(shares[c] for c in m.ALIGNMENT_CLASSES) == pytest.approx(1.0)
