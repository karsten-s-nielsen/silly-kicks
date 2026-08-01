"""Resume + bundle-fidelity for the TF-48 SB acceptance harness (ADR-052).

This driver is the hardest case in the migration: one match produces SIX heterogeneous outputs
(matched-shot rows, unmatched rows, a per-match report, sweep rows, a z-compare row, and the nested
``--debug-shots`` kernel capture), so its shard carries a JSON BUNDLE rather than a tidy table. Two
things therefore have to hold, and only the pair is evidence:

* a second run does no work, and
* the report it writes is IDENTICAL -- which is what proves the bundle lost none of the six.

Its only other coverage is `tests/tracking/test_shot_goalmouth_sb_e2e.py`, which is `@e2e` and
owner-token gated, i.e. it does not run here at all.
"""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

import scripts.validate_shot_goalmouth_sb as sbmod

_HOME_TEAM, _AWAY_TEAM = "A", "B"


def _sb_events() -> pd.DataFrame:
    gk = [{"teammate": False, "position": {"name": "Goalkeeper"}, "location": [118.0, 43.0]}]
    return pd.DataFrame(
        {
            "type": ["Shot", "Shot"],
            "period": [1, 1],
            "minute": [10, 20],
            "second": [0, 0],
            "team": ["Alpha", "Beta"],
            "shot_outcome": ["Goal", "Saved"],
            "shot_end_location": [[120.0, 36.0, 1.2], [120.0, 44.0, 0.9]],
            "shot_freeze_frame": [gk, gk],
        }
    )


def _enriched(spadlconfig) -> pd.DataFrame:
    """Two shot actions whose clocks line up with the stub SB events.

    The `pd.NA` on-target verdict is deliberate: it is the one value whose shard encoding is not
    obvious, and the harness asks `pd.isna` about it downstream.
    """
    shot = spadlconfig.actiontype_id["shot"]
    return pd.DataFrame(
        {
            "game_id": ["g", "g"],
            "action_id": [1, 2],
            "type_id": [shot, shot],
            "type_name": ["shot", "shot"],
            "result_name": ["success", "fail"],
            "period_id": [1, 1],
            "team_id": [_HOME_TEAM, _AWAY_TEAM],
            "time_seconds": [600.0, 1200.0],
            "start_x": [95.0, 90.0],
            "start_y": [30.0, 40.0],
            "shot_crossing_source": ["observed", "observed"],
            "shot_fit_end_reason": ["plane_straddle", "window_cap"],
            "shot_z_profile": ["airborne", "rolling"],
            "shot_crossing_y": [33.0, 36.0],
            "shot_crossing_z": [1.1, 0.8],
            "shot_on_target_derived": pd.array([True, pd.NA], dtype="boolean"),
        }
    )


def _frames() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g"] * 6,
            "period_id": [1] * 6,
            "frame_id": [1, 2, 3, 4, 5, 6],
            "time_seconds": [600.0, 600.0, 1200.0, 1200.0, 600.0, 1200.0],
            "team_id": [_HOME_TEAM, _AWAY_TEAM, _HOME_TEAM, _AWAY_TEAM, _HOME_TEAM, _AWAY_TEAM],
            "is_goalkeeper": [True, True, True, True, False, False],
            "is_ball": [False, False, False, False, True, True],
            "x": [5.0, 100.0, 5.0, 100.0, 50.0, 60.0],
            "y": [30.0, 38.0, 30.0, 38.0, 34.0, 34.0],
            "z": [0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
        }
    )


@pytest.fixture()
def stub_sb(monkeypatch):
    """Stub both remote sources and the enrichment; count entries into the real per-match work."""
    from silly_kicks.spadl import config as spadlconfig
    from silly_kicks.tracking import _gk_geometry, _gk_resolve, features

    entered: list[str] = []
    enriched, frames = _enriched(spadlconfig), _frames()

    fake = types.ModuleType("statsbombpy")
    fake.sb = types.SimpleNamespace(  # type: ignore[attr-defined]
        matches=lambda **_kw: pd.DataFrame(
            {"home_team": ["Alpha", "Gamma"], "away_team": ["Beta", "Delta"], "match_id": [7001, 7002]}
        ),
        events=lambda **_kw: _sb_events(),
    )
    monkeypatch.setitem(sys.modules, "statsbombpy", fake)

    manifest = [
        {"id": "1001", "date": "2022-11-21", "home": "Alpha", "away": "Beta"},
        # Names that map to NO stub SB team, so this match exercises the zero-matched-shot path --
        # its report and unmatched rows still have to survive the shard round trip.
        {"id": "1002", "date": "2022-11-22", "home": "Gamma", "away": "Delta"},
    ]
    monkeypatch.setattr(sbmod, "_resolve_token", lambda *_a, **_k: "tok")
    monkeypatch.setattr(sbmod, "_base_url", lambda *_a, **_k: "http://stub")
    monkeypatch.setattr(sbmod, "_list_matches", lambda *_a, **_k: manifest)

    def _load(**_kw):
        return iter([("gradientsports", m["id"], enriched, frames, _HOME_TEAM) for m in manifest])

    def _add(actions, frames_arg, **_kw):
        entered.append("enrich")
        return enriched.copy()

    monkeypatch.setattr(sbmod, "load_matches", _load)
    monkeypatch.setattr(features, "add_shot_goalmouth", _add)
    monkeypatch.setattr(
        _gk_resolve, "defended_goal_x", lambda _f: {("g", 1, _HOME_TEAM): 0.0, ("g", 1, _AWAY_TEAM): 105.0}
    )
    monkeypatch.setattr(_gk_geometry, "_truthy_bool", lambda s: s.astype(bool))
    return entered


def _run(tmp_path, stub, **kw) -> dict:
    return sbmod.run("all", str(tmp_path / "report.json"), None, None, **kw)


def test_sb_harness_second_run_does_NO_work(tmp_path, stub_sb):
    _run(tmp_path, stub_sb)
    first = len(stub_sb)
    assert first >= 2, "the first run should have enriched both matches"

    _run(tmp_path, stub_sb)

    assert len(stub_sb) == first, "the second run re-entered the enrichment"


def test_sb_harness_resumed_report_is_IDENTICAL(tmp_path, stub_sb):
    """The bundle-fidelity assertion. Every one of the six per-match outputs reaches the report
    through the shard, so an equal report is the statement that none of them was dropped."""
    before = _run(tmp_path, stub_sb, sweep=True)
    after = _run(tmp_path, stub_sb, sweep=True)

    assert json.dumps(after, sort_keys=True, default=str) == json.dumps(before, sort_keys=True, default=str)
    # ...and non-vacuously so: the report actually carries the side outputs being claimed.
    assert before["per_match_reports"] and before["sweep"] and before["unmatched"]
    assert before["n_matched"] > 0


def test_the_sb_resume_oracle_would_CATCH_a_lost_resume(tmp_path, stub_sb):
    """Without this, a green oracle is indistinguishable from one that never sharded anything."""
    _run(tmp_path, stub_sb)
    first = len(stub_sb)
    for shard in (tmp_path / "report_shards").rglob("*.parquet"):
        shard.unlink()

    _run(tmp_path, stub_sb)

    assert len(stub_sb) > first, "the oracle cannot detect a lost resume -- it is vacuous"


def test_an_optional_PASS_is_declared_so_it_cannot_reuse_a_pass_that_skipped_it(tmp_path, stub_sb):
    """`--sweep` / `--debug-shots` / `--tracking-limit` ADD records to the bundle. If they were not
    declared inputs, a `--sweep` run over a directory built without it would skip every match and
    report an EMPTY sweep -- a measurement that never ran, indistinguishable from one that found
    nothing."""
    plain = _run(tmp_path, stub_sb)
    assert plain["sweep"] == []
    entered_after_plain = len(stub_sb)

    swept = _run(tmp_path, stub_sb, sweep=True)

    assert len(stub_sb) > entered_after_plain, "the sweep run reused the non-sweep shards"
    assert swept["sweep"], "and it produced no sweep rows"


def test_the_shard_encoder_preserves_a_MISSING_on_target_verdict():
    """The bundle's one risky conversion, asserted from both sides.

    `default=str` -- the encoder the final report uses -- renders `pd.NA` as the string `"<NA>"`,
    and the debug block asks `pd.isna(r["on_target_derived"])` about exactly this field: NA means
    the crossing height is unknown, `"<NA>"` is a non-null string, so a shot would come back from a
    shard with a fabricated verdict.
    """
    rec = {
        "on_target_derived": pd.NA,
        "crossing_y_m": np.float64(33.5),
        "n": np.int64(3),
        "ok": np.bool_(True),
        "missing": float("nan"),
    }
    back = json.loads(json.dumps(rec, default=sbmod._json_default))

    assert back["on_target_derived"] is None
    assert pd.isna(back["on_target_derived"])
    assert back["crossing_y_m"] == 33.5
    assert back["n"] == 3
    assert back["ok"] is True
    assert np.isnan(back["missing"]), "a NaN measurement must not become None -- they differ here"

    stringified = json.loads(json.dumps(rec, default=str))
    assert not pd.isna(stringified["on_target_derived"]), "default=str is why this encoder exists"


def test_the_shard_encoder_REFUSES_an_undeclared_type():
    """Fail loud rather than stringify. A silent `str()` is how a value becomes plausible-looking
    and wrong -- the same reason `_settle_votes` records its fallback instead of taking it."""
    with pytest.raises(TypeError, match="no declared shard encoding"):
        json.dumps({"t": pd.Timestamp("2022-01-01")}, default=sbmod._json_default)
