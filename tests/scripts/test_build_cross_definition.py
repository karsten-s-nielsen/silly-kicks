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
    lines = m.CANDIDATE_LINES
    sb = dict(zip(lines, [0.60, 0.70, 0.66, 0.50], strict=True))
    gs = dict(zip(lines, [0.40, 0.50, 0.52, 0.30], strict=True))
    assert m.select_start_line({"statsbomb": sb, "gradientsports": gs}) == 60.0  # mean 0.60 beats 0.59
    # IDSSE is reported but not averaged.
    assert m.select_start_line({"statsbomb": sb, "gradientsports": gs, "idsse": dict.fromkeys(lines, 1.0)}) == 60.0


def test_selection_tie_picks_the_larger_line():
    lines = m.CANDIDATE_LINES
    sb = dict(zip(lines, [0.60, 0.700, 0.704, 0.50], strict=True))
    gs = dict(zip(lines, [0.40, 0.500, 0.500, 0.30], strict=True))
    assert m.select_start_line({"statsbomb": sb, "gradientsports": gs}) == 65.0  # within 0.005 of the best


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
    """Two providers, 20 applicable rows each. GS rows 0-2 are native crosses with an opponent first contact, and
    rows 0-1 are blocked, and row 1's first contact is a teammate: only row 0 passes every block-radius filter.
    Dropping the blocked or the opponent filter changes the sample (blocked implies native, so the native filter
    is redundant here)."""
    rows = []
    for provider, fr in (("gradientsports", 25.0), ("skillcorner", 10.0)):
        for i in range(20):
            rows.append(
                {
                    "provider": provider,
                    "applicable": True,
                    "release_source": "elastic",
                    "release_dist_m": float(i + 1) if provider == "gradientsports" else 0.5 * (i + 1),
                    "passer_contact_dt": 0.01 * (i + 1),
                    "first_contact_dist_m": 4.0 + i,
                    "first_contact_is_opponent": i != 1,
                    "cross_native": provider == "gradientsports" and i < 3,
                    "cross_blocked": provider == "gradientsports" and i < 2,
                    "coverage": 0.5 if i < 5 else 1.0,
                    "frame_rate": fr,
                }
            )
    df = pd.DataFrame(rows)
    for col in ("first_contact_is_opponent", "cross_native", "cross_blocked"):
        df[col] = df[col].astype("boolean")
    return df


def test_calibrate_applies_the_pre_declared_rules():
    rows = _calibration_rows()
    cal = m.calibrate(rows)
    by = {str(p): g["release_dist_m"].to_numpy() for p, g in rows.groupby("provider")}
    assert cal["release_match_m"] == m.weighted_quantile(by, 0.95)  # the equal-weight 95th percentile
    assert cal["release_match_m"] != m.weighted_quantile(by, 0.5)
    assert cal["block_radius_m"] == pytest.approx(4.0)  # the one GS blocked native cross
    assert cal["n_block_samples_gs"] == 1
    gaps = {str(p): g["passer_contact_dt"].to_numpy() for p, g in rows.groupby("provider")}
    assert cal["release_gap_s"] == pytest.approx(max(m.weighted_quantile(gaps, 0.95), 1 / 25.0))
    assert cal["direction_window_s"] == pytest.approx(2 / 10.0)  # two frames at the LOWEST frame rate
    assert cal["coverage_below_0.8_share"] == {"gradientsports": 0.25, "skillcorner": 0.25}
    assert cal["coverage_owner_decision_needed"] is True


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
        "evidence",
        "n_matches",
    }
    assert rep["reporting_line"] == 52.5 and rep["selected_start_line"] is None  # no StatsBomb arm in the toy
    ev = rep["evidence"]
    assert {"median_crosses_per_match", "ratio_bar", "cross_attempt", "alignment", "corner_kind_source"} <= set(ev)
    assert ev["alignment"], "precondition: the toy has applicable rows"
    for cells in ev["alignment"].values():
        assert set(cells) <= set(m.ROW_CLASSES)
        for shares in cells.values():
            assert sum(shares[c] for c in m.ALIGNMENT_CLASSES) == pytest.approx(1.0)
