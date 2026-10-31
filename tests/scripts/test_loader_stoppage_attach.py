"""Task 1 (TF-46 stoppage re-stage): the loader stoppage-attach seam.

The materializer re-stages the xShot/xCross corpora with ``stoppage=True`` so each match's frames carry
``stoppage_state``/``stoppage_source`` from :func:`silly_kicks.tracking.derive_stoppage_state`. The attach
must run on RAW full-rate frames BEFORE ``_preprocess`` (the ball-out start rule reads the ball's raw x/y).
``build_skillcorner_frames(preprocess=False)`` exposes those raw frames; the default path is byte-identical.

The end-to-end ``ball_state`` (IDSSE) / ``events_v3`` (Gradient Sports) attach through the loader flag is
pinned HERE by driving ``_build_idsse`` / ``_build_gradientsports`` with ``stoppage=True`` on the committed
fixtures (the seam tests below): reverting the attach sites to identity turns them red. The
``derive_stoppage_state`` behaviour itself is pinned in ``tests/tracking/test_stoppages.py``, and the
``for_each`` silent-shrink guard in ``tests/scripts/test_materialize_stoppage.py``. This file also pins the
raw-frame path and that ``_preprocess`` is additive (x/y are preserved).
"""

import json
import pathlib

import numpy as np
import pandas as pd
import pytest
from _loader_pining import build_skillcorner_frames

import scripts._loader_pining as lp
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking import STOPPAGE_SOURCES

_DS = pathlib.Path(__file__).resolve().parents[1] / "datasets"
_FX = _DS / "tracking" / "yident" / "skillcorner"

_EVENTS_V3 = STOPPAGE_SOURCES[1]
_IDSSE = _DS / "sportec" / "idsse_slice"
_IDSSE_PATHS = {"metadata": _IDSSE / "info.xml", "events": _IDSSE / "events.xml", "tracking": _IDSSE / "positions.xml"}
_IDSSE_MATCH = "DFL-MAT-J03WMX"

_GS_EVENTS = _DS / "gradientsports" / "synthetic_match.json"
_GS_ROSTER = [
    (100, "1", 1, "GK"),
    (100, "2", 2, "DEF"),
    (100, "7", 7, "MID"),
    (100, "9", 9, "FWD"),
    (100, "11", 11, "FWD"),
    (200, "12", 12, "GK"),
    (200, "20", 20, "FWD"),
]


def _write_gs_match(
    tmp: pathlib.Path,
    *,
    step_s: float = 1.0 / 30.0,
    n_frames: int = 30,
    with_visibility: bool = False,
    partial_visibility: bool = False,
) -> dict:
    """A tiny complete GS match (events + roster + metadata + a JSONL tracking file over periods 1-4), the
    events_only-test precedent, so ``_build_gradientsports(stoppage=True)`` runs the real events_v3 attach.

    ``step_s`` is the ball-sample spacing: the ~30 Hz default is full-rate (the attach's full-rate check
    passes); a coarse ``step_s`` (e.g. 0.5) is a THINNED track the check must refuse."""
    (tmp / "metadata.json").write_text(
        json.dumps(
            [
                {
                    "id": 99999,
                    "homeTeam": {"id": 100},
                    "awayTeam": {"id": 200},
                    "homeTeamStartLeft": True,
                    "homeTeamStartLeftExtraTime": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    (tmp / "roster.json").write_text(
        json.dumps(
            [
                {"team": {"id": t}, "shirtNumber": s, "player": {"id": p}, "positionGroupType": g}
                for t, s, p, g in _GS_ROSTER
            ]
        ),
        encoding="utf-8",
    )
    lines = []
    for period in (1, 2, 3, 4):
        gk_x = -45.0 if period in (1, 3) else 45.0
        for fr in range(n_frames):
            # `float | str`: the raw GS objects carry numeric coordinates and an OPTIONAL string
            # `visibility` flag; the union lets pyright accept both without widening to `object`.
            home: list[dict[str, float | str]] = [
                {"jerseyNum": 1, "x": gk_x, "y": 0.0},
                {"jerseyNum": 2, "x": gk_x * 0.6, "y": 5.0},
                {"jerseyNum": 7, "x": fr * 0.1, "y": -5.0},
                {"jerseyNum": 9, "x": -gk_x * 0.3, "y": 10.0},
                {"jerseyNum": 11, "x": -gk_x * 0.4, "y": -10.0},
            ]
            away: list[dict[str, float | str]] = [
                {"jerseyNum": 12, "x": -gk_x, "y": 0.0},
                {"jerseyNum": 20, "x": gk_x * 0.2, "y": 3.0},
            ]
            balls: list[dict[str, float | str]] = [{"x": fr * 0.2, "y": 0.0, "z": 0.1}]
            if with_visibility:
                # The raw GS per-object detection flag (VISIBLE / ESTIMATED). Mix the two for non-vacuity:
                # jersey 11 is ESTIMATED (undetected), the rest VISIBLE.
                for p in home + away:
                    p["visibility"] = "ESTIMATED" if p["jerseyNum"] == 11 else "VISIBLE"
                for b in balls:
                    b["visibility"] = "VISIBLE"
                if partial_visibility:
                    # One object carries NO flag -> a PARTIAL (not all-null) column the drop-guard keeps.
                    del away[0]["visibility"]
            lines.append(
                json.dumps(
                    {
                        "period": period,
                        "frameNum": period * 1000 + fr,
                        "periodGameClockTime": round(fr * step_s, 4),
                        "homePlayers": home,
                        "awayPlayers": away,
                        "balls": balls,
                    }
                )
            )
    (tmp / "tracking.jsonl").write_text("\n".join(lines), encoding="utf-8")
    (tmp / "events.json").write_bytes(_GS_EVENTS.read_bytes())
    return {
        "metadata": tmp / "metadata.json",
        "roster": tmp / "roster.json",
        "tracking": tmp / "tracking.jsonl",
        "events": tmp / "events.json",
    }


def _paths():
    return {"metadata": str(_FX / "match.json"), "tracking": str(_FX / "tracking_slice.jsonl")}


def test_preprocess_false_returns_raw_frames():
    """``preprocess=False`` returns the raw pre-smoothing frames (no ``x_smoothed``); the default path
    smooths. This is the raw surface the stoppage seam attaches on."""
    raw, _ = build_skillcorner_frames(_paths(), match_id="x", tracking_limit=None, preprocess=False)
    full, _ = build_skillcorner_frames(_paths(), match_id="x", tracking_limit=None, preprocess=True)
    assert len(raw) > 0
    assert "x_smoothed" not in raw.columns, "preprocess=False must return RAW frames (no smoothing columns)"
    assert "x_smoothed" in full.columns, "the default path must still smooth (byte-identical to before)"


def test_preprocess_is_additive_x_not_overwritten():
    """``_preprocess`` emits additive ``x_smoothed``/``y_smoothed`` and never overwrites ``x``/``y`` -- so a
    stoppage attach could read the ball's raw x/y even after preprocess; the seam attaches pre-preprocess
    regardless, but this pins the additivity premise."""
    full, _ = build_skillcorner_frames(_paths(), match_id="x", tracking_limit=None, preprocess=True)
    assert {"x", "y", "x_smoothed", "y_smoothed"} <= set(full.columns)
    assert not full["x"].equals(full["x_smoothed"]) or len(full) == 0  # smoothing produced a distinct column


# --- the loader stoppage-attach SEAM (Task 1): driving _build_*(stoppage=True) on committed fixtures ---
# B03: the toys pre-stamp via mark_all_live, so before this the loader attach (derive_stoppage_state behind
# the stoppage=True flag) had NO gate -- reverting all 3 attach sites to identity passed every test.


def test_flag_on_attaches_ball_state_for_sportec():
    actions, frames, _home = lp._build_idsse(_IDSSE_PATHS, _IDSSE_MATCH, None, stoppage=True)
    assert "stoppage_state" in frames.columns and "stoppage_source" in frames.columns
    assert set(frames["stoppage_source"].astype(str).unique()) == {"ball_state"}  # observed provider
    assert {"live", "dead"} <= set(frames["stoppage_state"].astype(str).unique())  # non-vacuous: real dead spans
    assert "t_kick" not in actions.columns  # the attach adds no t_kick to the actions


def test_flag_off_leaves_frames_without_stoppage_columns():
    _actions, frames, _home = lp._build_idsse(_IDSSE_PATHS, _IDSSE_MATCH, None, stoppage=False)
    assert "stoppage_state" not in frames.columns and "stoppage_source" not in frames.columns


def test_idsse_attach_and_preprocess_both_present():
    # The flag-on IDSSE frames carry BOTH the stoppage columns AND the preprocess smoothing columns -- the
    # attach ran and preprocess ran. This does NOT pin the ORDER (ball_state is read from the frame column,
    # not raw x/y, so order is unobservable here); the SkillCorner test below pins the raw-before-preprocess
    # order where it matters (the ball-out rule reads raw x/y).
    _actions, frames, _home = lp._build_idsse(_IDSSE_PATHS, _IDSSE_MATCH, None, stoppage=True)
    assert "stoppage_state" in frames.columns
    assert "x_smoothed" in frames.columns  # preprocess still ran


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_flag_on_attaches_events_v3_for_gradientsports(tmp_path):
    paths = _write_gs_match(tmp_path)
    actions, frames, _home = lp._build_gradientsports(paths, None, stoppage=True)
    assert "stoppage_state" in frames.columns and "stoppage_source" in frames.columns
    assert set(frames["stoppage_source"].astype(str).unique()) == {_EVENTS_V3}  # event-coded provider
    assert "t_kick" not in actions.columns


def test_flag_on_thinned_gs_frames_raises(tmp_path):
    # plan L57: a thinned (not full-rate) ball track makes derive_stoppage_state raise its full-rate check;
    # the staging seam must SURFACE it, never write a shard with a wrong dead-ball domain.
    paths = _write_gs_match(tmp_path, step_s=0.5, n_frames=20)
    with pytest.raises(ValueError, match="full-rate"):
        lp._build_gradientsports(paths, None, stoppage=True)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_gradientsports_pining_carries_the_detection_flag(tmp_path):
    # GS is detection-aware (ADR-069 amended 2026-10-26): `_build_gradientsports` must plumb the raw
    # per-object VISIBLE / ESTIMATED flag into `visibility`, so a real feed does not fail the detection
    # gate. Without the flag (every other GS test here) the all-null column is dropped and `visibility`
    # is None -- the legacy path, which keeps those tests green.
    paths = _write_gs_match(tmp_path, with_visibility=True)
    _actions, frames, _home = lp._build_gradientsports(paths, None)
    assert frames["visibility"].notna().any()  # the flag reached the frames
    # Non-vacuous: jersey 11 is ESTIMATED (False), the rest VISIBLE (True).
    assert set(frames["visibility"].dropna().astype(bool).tolist()) == {True, False}


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_gradientsports_pining_partial_visibility_fails_loud(tmp_path):
    # B20: a PARTIAL detection flag (one object has no flag) is NOT all-null, so the drop-guard keeps the
    # column and the closed-vocab check in convert_to_frames rejects the null -- fail loud, never a silent
    # half-detected feed. Pins the `.isna().all()` boundary: an `.any()` mutant would drop the column and
    # yield visibility=None with no error.
    paths = _write_gs_match(tmp_path, with_visibility=True, partial_visibility=True)
    with pytest.raises(ValueError, match="visibility_native"):
        lp._build_gradientsports(paths, None)


# --- the SkillCorner attach site (_build_skillcorner, the :1037 site) + the raw-before-preprocess ORDER ---
# B03: reverting the SkillCorner attach site alone was uncaught, and no test pinned the order. The frames +
# actions mirror tests/tracking/test_stoppages.py's known-valid events_v3 fixture; the native SkillCorner
# event+tracking parse is stubbed so these need no full match fixture.


def _sc_ball_frames() -> pd.DataFrame:
    t = np.round(np.arange(9.0, 52.0, 0.1), 3)
    return pd.DataFrame(
        {
            "game_id": 7,
            "period_id": 1,
            "frame_id": np.arange(len(t)),
            "time_seconds": t,
            "frame_rate": 10.0,
            "is_ball": True,
            "x": 50.0,
            "y": np.where((t >= 12.0 - 1e-9) & (t < 20.0), -0.5, 30.0),  # ball out 12-20 s -> a dead span
            "visibility": True,
            "ball_state": "alive",
            "source_provider": "skillcorner",
        }
    )


def _sc_actions(*, with_gi: bool) -> pd.DataFrame:
    def act(aid, tt, name, provenance):
        row = {
            "game_id": 7,
            "period_id": 1,
            "action_id": aid,
            "time_seconds": float(tt),
            "team_id": "A",
            "type_id": spadlconfig.actiontypes.index(name),
            "result_id": spadlconfig.results.index("success"),
            "action_provenance": provenance,
        }
        if with_gi:  # the current converter carries the GI tag; its absence is the old-converter raise (RF2)
            row["gi_interruption_after"] = pd.NA
            row["gi_possession_frame_end"] = pd.NA
        return row

    return pd.DataFrame(
        [
            act(1, 10.0, "pass", "native"),
            act(2, 25.0, "throw_in", "derived"),
            act(3, 40.0, "pass", "native"),
            act(4, 50.0, "corner_crossed", "derived"),
        ]
    )


def test_skillcorner_flag_on_attaches_events_v3_on_raw_frames(monkeypatch):
    # The SkillCorner attach site (:1037) specifically + the ORDER: _build_skillcorner builds RAW frames
    # (preprocess=False), runs derive_stoppage_state (events_v3), THEN _preprocess. Reverting the attach to
    # identity, or moving it after _preprocess (build_skillcorner_frames called preprocess=True), breaks
    # this. The native parse is stubbed; _preprocess is identity to isolate the attach.
    frames = _sc_ball_frames()
    calls: dict = {}

    def spy_build(paths, match_id, tracking_limit, *, with_velocity=True, preprocess=True):
        calls["preprocess"] = preprocess
        return frames.copy(), None

    monkeypatch.setattr(lp, "build_skillcorner_frames", spy_build)
    monkeypatch.setattr(lp, "_skillcorner_actions", lambda paths: (_sc_actions(with_gi=True), "home"))
    monkeypatch.setattr(lp, "_preprocess", lambda f, with_velocity=True: f)
    _a, out, _h, _r = lp._build_skillcorner({}, "x", None, stoppage=True)
    assert calls["preprocess"] is False  # raw frames built BEFORE _preprocess (spec section 4.1)
    assert "stoppage_state" in out.columns
    assert set(out["stoppage_source"].astype(str).unique()) == {_EVENTS_V3}  # the :1037 attach fired
    states = set(out["stoppage_state"].astype(str).unique())
    assert "live" in states and states - {"live"}  # non-vacuous: the restart/ball-out spans gave a non-live state


def test_skillcorner_flag_on_stale_actions_raises(monkeypatch):
    # RF2 / plan L58: SkillCorner actions without gi_interruption_after come from an old converter, so
    # derive_stoppage_state raises; the staging seam must SURFACE it, never write a shard.
    monkeypatch.setattr(lp, "build_skillcorner_frames", lambda *a, **k: (_sc_ball_frames(), None))
    monkeypatch.setattr(lp, "_skillcorner_actions", lambda paths: (_sc_actions(with_gi=False), "home"))
    with pytest.raises(ValueError, match="reconvert"):
        lp._build_skillcorner({}, "x", None, stoppage=True)
