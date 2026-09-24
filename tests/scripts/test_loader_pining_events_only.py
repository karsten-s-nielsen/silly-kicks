"""The pining loader's EVENTS-ONLY path (TF-56 Commit-2 driver xT-fit scaling fix).

xT is event-only: ``ExpectedThreat.fit`` reads SPADL actions and never tracking. The TF-56 driver used
to obtain its fit corpus via ``list(load_matches(..., tracking_limit=0))``, which (a) still DOWNLOADED
and PARSED every match's tracking (``tracking_limit`` caps frames only AFTER the parse; measured
~44 s/match, ~12 h over the 980-match owner corpus) and (b) materialized every match's frames at once
(the DGX fit process died of OOM with no traceback). ``events_only=True`` never requests the tracking
artifact and never builds frames, so the full corpus streams in minutes with bounded memory.

The load-bearing property is BYTE-IDENTITY of the actions: an events-only build must produce exactly
the actions the full build produces, or the xT surface would silently depend on which path fit it.
Each builder is proven against its full path on committed/real-shaped fixtures -- with the tracking
artifact ABSENT from ``paths``, so any read of it would raise rather than pass unnoticed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import scripts._loader_pining as lp

_DATASETS = Path(__file__).resolve().parents[1] / "datasets"
_IDSSE = _DATASETS / "sportec" / "idsse_slice"
_SC = _DATASETS / "skillcorner"
_GS_EVENTS = _DATASETS / "gradientsports" / "synthetic_match.json"
_SB360 = _DATASETS / "statsbomb" / "three-sixty"  # WWC2023 golden slice (events.json + frames.json)
_SB360_MATCH_ID = 3893795

_SC_ARTIFACTS = {
    "a": "1886347_dynamic_events.csv",
    "b": "1886347_match.json",
    "c": "1886347_tracking_extrapolated.jsonl",
}


# --------------------------------------------------------------------------------------------------
# _artifact_roles: which artifacts a build downloads
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "artifacts"),
    [("idsse", {}), ("gradientsports", {}), ("skillcorner", _SC_ARTIFACTS)],
)
def test_artifact_roles_request_tracking_only_on_the_full_path(provider, artifacts):
    full = lp._artifact_roles(provider, artifacts)
    events_only = lp._artifact_roles(provider, artifacts, events_only=True)
    assert "tracking" in full
    assert "tracking" not in events_only  # the whole point: the tracking file is never downloaded
    assert {"events", "metadata"} <= set(events_only)
    assert set(events_only) == set(full) - {"tracking"}  # nothing else is dropped


def test_artifact_roles_skillcorner_events_only_does_not_need_a_tracking_artifact():
    # Resolving the tracking key raises KeyError when the manifest lists none; events-only must not ask.
    no_tracking = {k: v for k, v in _SC_ARTIFACTS.items() if "tracking" not in v}
    with pytest.raises(KeyError):
        lp._artifact_roles("skillcorner", no_tracking)
    roles = lp._artifact_roles("skillcorner", no_tracking, events_only=True)
    assert roles == {"events": "a", "metadata": "b"}


def test_artifact_roles_statsbomb_events_only_drops_freeze_frames(monkeypatch):
    # INVERTED (CDLS-SPEC-03, ratified 2026-09-23): freeze_frames IS the tracking-side artifact and the
    # actions never read it, so an events-only StatsBomb load drops it exactly as the tracking providers
    # drop `tracking`. It is not a caller error.
    full = lp._artifact_roles("statsbomb", {})
    events_only = lp._artifact_roles("statsbomb", {}, events_only=True)
    assert "freeze_frames" in full
    assert "freeze_frames" not in events_only
    assert set(events_only) == set(full) - {"freeze_frames"} == {"events", "metadata", "roster"}


def test_artifact_roles_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown pining provider"):
        lp._artifact_roles("nope", {})


# --------------------------------------------------------------------------------------------------
# Per-provider byte-identity: the events-only build (_build_match_actions) == the full build's actions,
# with the tracking artifact ABSENT from paths (any tracking read raises KeyError)
# --------------------------------------------------------------------------------------------------


def test_idsse_events_only_actions_are_byte_identical_to_the_full_build():
    full_paths = {
        "metadata": _IDSSE / "info.xml",
        "events": _IDSSE / "events.xml",
        "tracking": _IDSSE / "positions.xml",
    }
    events_paths = {k: v for k, v in full_paths.items() if k != "tracking"}
    full_actions, full_frames, full_home = lp._build_idsse(full_paths, "DFL-MAT-J03WMX", None)
    actions, home = lp._build_match_actions("idsse", "DFL-MAT-J03WMX", events_paths)
    assert home == full_home
    assert len(full_frames) > 0  # non-vacuity: the full build really did build frames
    pd.testing.assert_frame_equal(actions, full_actions)


def test_skillcorner_events_only_builds_actions_without_tracking():
    from silly_kicks.spadl import skillcorner as sk_spadl

    paths = {"metadata": _SC / "match_metadata.json", "events": _SC / "basic_possessions.csv"}
    actions, home = lp._build_match_actions("skillcorner", "sc-test", paths)
    meta = json.loads((_SC / "match_metadata.json").read_text(encoding="utf-8"))
    assert home == str(meta["home_team"]["id"])
    expected, _ = sk_spadl.convert_to_actions(pd.read_csv(_SC / "basic_possessions.csv", low_memory=False), meta)
    assert len(expected) > 0  # non-vacuity
    pd.testing.assert_frame_equal(actions, expected)


_GS_ROSTER = [
    (100, "1", 1, "GK"),
    (100, "2", 2, "DEF"),
    (100, "7", 7, "MID"),
    (100, "9", 9, "FWD"),
    (100, "11", 11, "FWD"),
    (200, "12", 12, "GK"),
    (200, "20", 20, "FWD"),
]


def _write_gs_match(tmp: Path, *, et_flag) -> dict[str, Path]:
    """A tiny but complete GS match: the committed synthetic events (periods 1-4, i.e. WITH extra
    time) + metadata/roster + a small JSONL tracking file spanning the same four periods."""
    meta = [
        {
            "id": 99999,
            "homeTeam": {"id": 100},
            "awayTeam": {"id": 200},
            "homeTeamStartLeft": True,
            "homeTeamStartLeftExtraTime": et_flag,
        }
    ]
    (tmp / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    roster = [
        {"team": {"id": t}, "shirtNumber": s, "player": {"id": p}, "positionGroupType": g} for t, s, p, g in _GS_ROSTER
    ]
    (tmp / "roster.json").write_text(json.dumps(roster), encoding="utf-8")
    lines = []
    for period in (1, 2, 3, 4):
        gk_x = -45.0 if period in (1, 3) else 45.0  # home keeper defends the left goal in periods 1 and 3
        for f in range(20):
            home = [
                {"jerseyNum": 1, "x": gk_x, "y": 0.0},
                {"jerseyNum": 2, "x": gk_x * 0.6, "y": 5.0},
                {"jerseyNum": 7, "x": f * 0.1, "y": -5.0},
                {"jerseyNum": 9, "x": -gk_x * 0.3, "y": 10.0},
                {"jerseyNum": 11, "x": -gk_x * 0.4, "y": -10.0},
            ]
            away = [{"jerseyNum": 12, "x": -gk_x, "y": 0.0}, {"jerseyNum": 20, "x": gk_x * 0.2, "y": 3.0}]
            lines.append(
                json.dumps(
                    {
                        "period": period,
                        "frameNum": period * 1000 + f,
                        "periodGameClockTime": f * 0.5,
                        "homePlayers": home,
                        "awayPlayers": away,
                        "balls": [{"x": f * 0.2, "y": 0.0, "z": 0.1}],
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


@pytest.mark.filterwarnings("ignore::UserWarning")  # the documented ET-dropped warning (et-unknown leg)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # the pre-existing omitted output_convention (ADR-006)
@pytest.mark.parametrize("et_flag", [True, False, None], ids=["et-left", "et-right", "et-unknown"])
def test_gradientsports_events_only_actions_are_byte_identical_to_the_full_build(tmp_path, et_flag):
    """GS used to resolve the extra-time direction inside its FRAME block and reuse it for the events, so
    the events-only path must reach the SAME resolution without tracking -- including the ``None`` case,
    where extra time is DROPPED from both actions and frames on the full path."""
    full_paths = _write_gs_match(tmp_path, et_flag=et_flag)
    events_paths = {k: v for k, v in full_paths.items() if k != "tracking"}
    full_actions, full_frames, full_home = lp._build_gradientsports(full_paths, None)
    actions, home = lp._build_match_actions("gradientsports", "99999", events_paths)
    assert home == full_home
    assert len(full_frames) > 0  # non-vacuity: the full build really did build frames
    pd.testing.assert_frame_equal(actions, full_actions)
    # non-vacuity on the ET branch: the fixture really carries extra time, and None really drops it.
    et_present = set(full_actions["period_id"].unique()) & {3, 4}
    assert bool(et_present) is (et_flag is not None)


# --------------------------------------------------------------------------------------------------
# Thread-through: load_matches -> _build_match_with_retry -> _download_artifacts + _build_match_actions
# --------------------------------------------------------------------------------------------------


def test_statsbomb_events_only_actions_are_byte_identical_to_the_full_build():
    # INVERTED (CDLS-SPEC-03): StatsBomb events-only actions == the full build's actions, with the
    # freeze_frames artifact ABSENT from paths (any read of it would KeyError, not pass unnoticed).
    full_paths = {"events": _SB360 / "events.json", "freeze_frames": _SB360 / "frames.json"}
    events_paths = {"events": _SB360 / "events.json"}
    full_actions, full_frames, _home, _va, _report = lp.build_statsbomb_match(full_paths, _SB360_MATCH_ID)
    actions, home = lp._build_match_actions("statsbomb", _SB360_MATCH_ID, events_paths)
    assert len(full_frames) > 0  # non-vacuity: the full build really did build snapshot frames
    pd.testing.assert_frame_equal(actions, full_actions)
    assert home is not None


def test_build_match_actions_refuses_an_unknown_provider():
    with pytest.raises(ValueError, match="unknown pining provider"):
        lp._build_match_actions("nope", "m", {})


def _no_full_build(*_a, **_k):
    raise AssertionError("the events-only path must never reach the full (tracking) build")


def test_load_match_events_only_yields_frames_none_and_the_full_build_actions(monkeypatch):
    """load_match(events_only=True) end to end (offline; the wrappers no longer carry events_only per
    CDLS-SPEC-29): it requests no tracking artifact, never reaches the full build, and returns
    frames=None with actions byte-identical to the full build's."""
    full_actions, full_frames, _home = lp._build_idsse(
        {"metadata": _IDSSE / "info.xml", "events": _IDSSE / "events.xml", "tracking": _IDSSE / "positions.xml"},
        "DFL-MAT-J03WMX",
        None,
    )
    assert len(full_frames) > 0  # non-vacuity: the full build really did build frames

    def _fake_download(provider, match_id, artifacts, token, base_url, tmp_dir, *, use_cache=False, events_only=False):
        assert events_only is True and "tracking" not in lp._artifact_roles(provider, artifacts, events_only=True)
        return {"metadata": _IDSSE / "info.xml", "events": _IDSSE / "events.xml"}

    monkeypatch.setattr(lp, "_download_artifacts", _fake_download)
    monkeypatch.setattr(lp, "_build_match", _no_full_build)  # events-only must never reach the full build

    m = lp.load_match(lp.MatchRef("idsse", "DFL-MAT-J03WMX", {}), events_only=True)
    assert m.frames is None and m.visible_area is None and m.report is None
    pd.testing.assert_frame_equal(m.actions, full_actions)


def test_load_matches_default_is_unchanged_full_path(monkeypatch):
    """Hyrum: every existing caller omits events_only and must keep the full (tracking) build."""
    seen: list = []
    monkeypatch.setattr(lp, "_resolve_token", lambda token: "tok")
    monkeypatch.setattr(lp, "_base_url", lambda: "http://x")
    monkeypatch.setattr(lp, "_list_matches", lambda provider, tok, base: [{"id": "m1", "artifacts": {}}])

    def _fake_retry(provider, match_id, artifacts, tok, base_url, tracking_limit, *, cache_dir=None, events_only=False):
        seen.append(events_only)
        return pd.DataFrame(), pd.DataFrame({"f": [1]}), "H", None, None

    monkeypatch.setattr(lp, "_build_match_with_retry", _fake_retry)
    out = list(lp.load_matches(providers=["idsse"]))
    assert seen == [False]
    assert len(out) == 1 and out[0][3] is not None


def test_build_match_with_retry_threads_events_only_on_both_paths(monkeypatch, tmp_path):
    """Both the persistent-cache path and the temp-dir path route events-only to the actions build."""
    seen: list = []

    def _fake_download(provider, match_id, artifacts, token, base_url, tmp_dir, *, use_cache=False, events_only=False):
        seen.append(("download", use_cache, events_only))
        return {}

    def _fake_actions(provider, match_id, paths):
        seen.append(("actions", match_id))
        return "A", "H"

    monkeypatch.setattr(lp, "_download_artifacts", _fake_download)
    monkeypatch.setattr(lp, "_build_match_actions", _fake_actions)
    monkeypatch.setattr(lp, "_build_match", _no_full_build)
    expected = ("A", None, "H", None, None)  # the uniform 5-tuple, no frames / visible area / report
    assert (
        lp._build_match_with_retry("idsse", "m", {}, "t", "u", None, cache_dir=tmp_path, events_only=True) == expected
    )
    assert lp._build_match_with_retry("idsse", "m", {}, "t", "u", None, events_only=True) == expected
    assert seen == [("download", True, True), ("actions", "m"), ("download", False, True), ("actions", "m")]
