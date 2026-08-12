"""TF-35 run DETECTION ground truth (ADR-042).

Hand-written oracle. One game, team 1 home (attacks "ltr"), team 2 away ("rtl").
Every runner below is designed BY CONSTRUCTION, so neither implementation is
pinned to the other -- ``detect_off_ball_runs`` and the TF-4 kernel are both
checked against the SAME external truth (spec R2-2).

Action 10 (home pass by pid 1, t=30.0), window = the preceding 1.5 s:
  A  pid 101: displacement 6.0 m, peak 7.0 m/s   -> qualifies sprint-ON and sprint-OFF
  B  pid 102: displacement 4.0 m, peak 3.5 m/s   -> qualifies sprint-OFF only
  C  pid 103: displacement 2.0 m, peak 8.0 m/s   -> fails displacement, never qualifies
  GK pid 100 (is_goalkeeper): 6.0 m, fast        -> excluded by candidacy
  Opp pid 201: 8.0 m, fast                       -> excluded (other team)
  D  pid 104: a single frame in the window       -> excluded (<2 frames)
Action 11 (home pass, t=60.0): ball_state "dead" at the action frame -> NO rows.
Action 12 (AWAY pass by team 2, t=90.0): runner E pid 202 covers 6.0 m at 7.0 m/s
  toward frame x=0 (the goal the away team attacks) -> the emitted positions must be
  ACTION-LTR, so the run reads as forward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import features as F
from silly_kicks.tracking._run_values import _FLOOR_BY_METHOD, RunValuationParams, detect_off_ball_runs

_HOME, _AWAY = 1, 2
_PRE = 1.5
#: Window sample times as offsets from the action clock.
_OFFSETS = (-1.5, -1.0, -0.5, 0.0)

EXPECTED_SPRINT_OFF = {(10, 101), (10, 102), (12, 202)}
EXPECTED_SPRINT_ON = {(10, 101), (12, 202)}

#: TF-4 kernel counts on the SAME hand-written truth: 2 runners / dead / 1.
EXPECTED_TF4_COUNTS = {10: 2, 11: pd.NA, 12: 1}


def _frow(pid, team, gk, x, y, t, *, speed, is_ball=False, ball_state="alive"):
    return {
        "game_id": 1,
        "period_id": 1,
        "frame_id": round(t * 10),
        "time_seconds": float(t),
        "frame_rate": 10.0,
        "player_id": pid,
        "team_id": team,
        "is_ball": is_ball,
        "is_goalkeeper": gk,
        "x": float(x),
        "y": float(y),
        "z": 0.0,
        "speed": speed,
        "vx": 0.0,
        "vy": 0.0,
        "speed_source": "native",
        "ball_state": ball_state,
        "team_attacking_direction": None if is_ball else ("ltr" if team == _HOME else "rtl"),
        "confidence": None,
        "visibility": None,
        "source_provider": "synthetic",
        "is_goalkeeper_source": "native",
    }


#: The window offset the peak-speed sample lands on. Must be one of ``_OFFSETS`` --
#: an earlier draft keyed the peak on ``frac == 0.5``, which no sample ever hits, so
#: every runner silently carried only its half-speed samples and the sprint gate
#: emptied out.
_PEAK_AT_OFFSET = -0.5


def _movers_for_action_10(off: float, frac: float, t: float, rows: list) -> None:
    """frac in [0, 1] along the window. All displacements are pure +x."""
    # (pid, team, gk, x0, dx, peak_speed)
    spec = [
        (101, _HOME, False, 40.0, 6.0, 7.0),
        (102, _HOME, False, 40.0, 4.0, 3.5),
        (103, _HOME, False, 40.0, 2.0, 8.0),
        (100, _HOME, True, 10.0, 6.0, 7.0),
        (201, _AWAY, False, 70.0, 8.0, 7.0),
    ]
    for pid, team, gk, x0, dx, peak in spec:
        # Peak lands on one real sample; the rest sit below it.
        speed = peak if off == _PEAK_AT_OFFSET else peak * 0.5
        rows.append(_frow(pid, team, gk, x0 + dx * frac, 34.0, t, speed=speed))


def _frames(*, dead_at_action_11: bool = True) -> pd.DataFrame:
    rows: list = []

    # --- Action 10 window (t = 28.5 .. 30.0) ---
    for off in _OFFSETS:
        t = 30.0 + off
        frac = (off + _PRE) / _PRE
        rows.append(_frow(None, None, False, 52.5, 34.0, t, speed=8.0, is_ball=True))
        _movers_for_action_10(off, frac, t, rows)
    # D: present only at the action frame -> a single row, cannot form a run.
    rows.append(_frow(104, _HOME, False, 44.0, 20.0, 30.0, speed=7.0))

    # --- Action 11 window (t = 58.5 .. 60.0): ball dead at the action frame ---
    for off in _OFFSETS:
        t = 60.0 + off
        frac = (off + _PRE) / _PRE
        state = "dead" if (dead_at_action_11 and off == 0.0) else "alive"
        rows.append(_frow(None, None, False, 52.5, 34.0, t, speed=0.0, is_ball=True, ball_state=state))
        for pid in (101, 102):
            rows.append(_frow(pid, _HOME, False, 40.0 + 6.0 * frac, 34.0, t, speed=7.0))

    # --- Action 12 window (t = 88.5 .. 90.0): AWAY runner toward frame x=0 ---
    for off in _OFFSETS:
        t = 90.0 + off
        frac = (off + _PRE) / _PRE
        rows.append(_frow(None, None, False, 52.5, 34.0, t, speed=8.0, is_ball=True))
        speed = 7.0 if off == _PEAK_AT_OFFSET else 3.5
        rows.append(_frow(202, _AWAY, False, 60.0 - 6.0 * frac, 34.0, t, speed=speed))
        rows.append(_frow(203, _AWAY, False, 20.0, 50.0, t, speed=0.0))  # stationary teammate

    return pd.DataFrame(rows)


def _actions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "action_id": [10, 11, 12],
            "period_id": [1, 1, 1],
            "time_seconds": [30.0, 60.0, 90.0],
            "team_id": pd.Series([_HOME, _HOME, _AWAY], dtype="int64"),
            "player_id": pd.Series([1, 1, 2], dtype="int64"),
            "start_x": [52.5, 52.5, 52.5],
            "start_y": [34.0, 34.0, 34.0],
            "end_x": [80.0, 60.0, 80.0],
            "end_y": [34.0, 34.0, 34.0],
            "type_id": [0, 0, 0],
            "type_name": ["pass", "pass", "pass"],
            "result_id": [1, 1, 1],
            "result_name": ["success", "success", "success"],
            "bodypart_id": [0, 0, 0],
            "bodypart_name": ["foot", "foot", "foot"],
        }
    )


def _run_row(runs: pd.DataFrame, action_id: int, player_id: int) -> dict:
    """One run as a plain dict.

    ``runs.set_index([...]).loc[(a, p)]`` is typed ``Series | DataFrame | Scalar`` by
    pandas-stubs, so every subsequent ``row["col"]`` fails type checking. Returning a dict
    keeps the assertions readable AND type-clean, and the length assertion makes an
    ambiguous match fail loudly instead of silently indexing the first row.
    """
    match = runs[(runs["action_id"] == action_id) & (runs["player_id"] == player_id)]
    assert len(match) == 1, f"expected exactly 1 run for ({action_id}, {player_id}), got {len(match)}"
    return match.iloc[0].to_dict()


def _pairs(runs: pd.DataFrame) -> set[tuple[int, int]]:
    return {(int(a), int(p)) for a, p in zip(runs["action_id"], runs["player_id"], strict=True)}


class TestDetectionMatchesHandWrittenTruth:
    def test_sprint_gate_off_reproduces_the_oracle(self):
        runs = detect_off_ball_runs(_actions(), _frames(), params=RunValuationParams(min_peak_speed_ms=0.0))
        assert _pairs(runs) == EXPECTED_SPRINT_OFF

    def test_sprint_gate_on_reproduces_the_oracle(self):
        """Discriminator: B (3.5 m/s) drops out, A and E survive."""
        runs = detect_off_ball_runs(_actions(), _frames())
        assert _pairs(runs) == EXPECTED_SPRINT_ON
        assert EXPECTED_SPRINT_ON < EXPECTED_SPRINT_OFF, "the two gates must differ or the test is vacuous"

    def test_tf4_kernel_matches_the_same_oracle(self):
        """Both implementations answer to the hand-written truth, not to each other."""
        out = F.add_off_ball_runs(_actions(), _frames(), home_team_id=_HOME)
        counts = dict(zip(_actions()["action_id"], out["n_off_ball_runners_pre_window"], strict=True))
        assert counts[10] == EXPECTED_TF4_COUNTS[10]
        assert pd.isna(counts[11])
        assert counts[12] == EXPECTED_TF4_COUNTS[12]

    def test_dead_ball_action_emits_no_runs(self):
        runs = detect_off_ball_runs(_actions(), _frames(), params=RunValuationParams(min_peak_speed_ms=0.0))
        assert 11 not in set(runs["action_id"]), "a dead-ball action must not emit runs"
        assert len(detect_off_ball_runs(_actions(), _frames(dead_at_action_11=False)).query("action_id == 11")) > 0, (
            "with the ball alive the same action DOES emit runs - the dead-ball assertion is not vacuous"
        )

    def test_measured_geometry_is_the_hand_written_geometry(self):
        runs = detect_off_ball_runs(_actions(), _frames(), params=RunValuationParams(min_peak_speed_ms=0.0))
        a = _run_row(runs, 10, 101)
        assert float(a["displacement_m"]) == pytest.approx(6.0)
        assert float(a["duration_s"]) == pytest.approx(1.5)
        assert float(a["mean_speed_ms"]) == pytest.approx(4.0)
        assert float(a["peak_speed_ms"]) == pytest.approx(7.0)
        assert a["peak_speed_source"] == "measured"


class TestActionLtrEmission:
    def test_away_run_positions_are_action_ltr(self):
        """E advances toward frame x=0, which IS forward for the away team."""
        runs = detect_off_ball_runs(_actions(), _frames())
        e = _run_row(runs, 12, 202)
        assert float(e["run_start_x"]) == pytest.approx(105.0 - 60.0)
        assert float(e["run_end_x"]) == pytest.approx(105.0 - 54.0)
        assert float(e["run_end_x"]) > float(e["run_start_x"])
        assert bool(e["toward_goal"]) is True

    def test_home_run_positions_are_unflipped(self):
        runs = detect_off_ball_runs(_actions(), _frames())
        a = _run_row(runs, 10, 101)
        assert float(a["run_start_x"]) == pytest.approx(40.0)
        assert float(a["run_end_x"]) == pytest.approx(46.0)
        assert bool(a["toward_goal"]) is True


class TestSpeedFallback:
    def test_all_nan_speed_falls_back_to_displacement_rate(self):
        frames = _frames()
        frames["speed"] = np.nan
        runs = detect_off_ball_runs(_actions(), frames, params=RunValuationParams(min_peak_speed_ms=3.0))
        a = _run_row(runs, 10, 101)
        assert a["peak_speed_source"] == "displacement_rate"
        assert float(a["peak_speed_ms"]) == pytest.approx(4.0)  # 6.0 m / 1.5 s

    def test_measured_speed_is_preferred_when_present(self):
        runs = detect_off_ball_runs(_actions(), _frames(), params=RunValuationParams(min_peak_speed_ms=0.0))
        assert set(runs["peak_speed_source"]) == {"measured"}


class TestFloorResolution:
    def test_floor_resolution_fail_loud(self):
        assert RunValuationParams().resolved_region_floor() == _FLOOR_BY_METHOD["spearman"]
        with pytest.raises(ValueError, match="no calibrated floor"):
            RunValuationParams(pitch_control_method="voronoi").resolved_region_floor()
        assert (
            RunValuationParams(pitch_control_method="voronoi", region_influence_floor=0.5).resolved_region_floor()
            == 0.5
        )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"pre_seconds": 0.0}, "pre_seconds"),
            ({"min_displacement_m": -1.0}, "min_displacement_m"),
            ({"min_peak_speed_ms": -0.1}, "min_peak_speed_ms"),
            ({"region_influence_floor": 1.5}, "region_influence_floor"),
        ],
    )
    def test_invalid_params_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            RunValuationParams(**kwargs)


class TestEmptyInputs:
    def test_no_actions_returns_empty_with_schema(self):
        runs = detect_off_ball_runs(_actions().iloc[:0], _frames())
        assert len(runs) == 0
        assert "run_value" not in runs.columns
        for col in ("action_id", "player_id", "run_start_x", "toward_goal", "peak_speed_source"):
            assert col in runs.columns

    def test_no_frames_returns_empty(self):
        assert len(detect_off_ball_runs(_actions(), _frames().iloc[:0])) == 0
