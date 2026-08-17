"""Tests for the pining ``statsbomb`` loader path (``scripts/_loader_pining.py``).

The heavy chain (raw events -> SPADL actions -> freeze-frame snapshots -> tracking frames +
visible_area) runs against the committed, redistributable open-360 golden slice at
``tests/datasets/statsbomb/three-sixty/`` (WWC2023, 6 freeze-frames). The network-only pining
fetch is exercised by an owner-run probe, not here.

The public ``load_matches`` 5-tuple contract and the parallel ``load_statsbomb_matches`` 6-tuple
are checked with a monkeypatched build (no network).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import scripts._loader_pining as L

_360 = Path(__file__).resolve().parents[1] / "datasets" / "statsbomb" / "three-sixty"
_MATCH_ID = 3893795  # the golden slice's real match id (see tests/datasets/statsbomb/README.md)


def _events() -> list[dict]:
    return json.loads((_360 / "events.json").read_text(encoding="utf-8"))


def _home_from_events() -> int:
    return int(next(e["team"]["id"] for e in _events() if e.get("team")))


def _write(tmp: Path, name: str, obj) -> Path:
    p = tmp / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def test_build_statsbomb_match_shapes():
    paths = {"events": _360 / "events.json", "freeze_frames": _360 / "frames.json"}
    actions, frames, home, visible_area, report = L.build_statsbomb_match(paths, _MATCH_ID)

    assert len(actions) > 0
    assert home is not None
    # Freeze-frames carry no per-player temporal history -> speed is structurally unavailable.
    assert (frames["speed_source"] == "unavailable").all()
    # visible_area is one polygon per action that had a frame.
    assert {"action_id", "polygon"}.issubset(visible_area.columns)
    assert len(visible_area) > 0
    assert report is None


def test_build_statsbomb_match_threads_fidelity(tmp_path):
    """The metadata fidelity version is THREADED into ``convert_to_actions``.

    Proven via the converter's own inference warning rather than a coordinate diff: for this
    fixture fidelity 1 and 2 happen to produce identical coordinates (a converter/data property,
    verified at the ``convert_to_actions`` seam), so the honest, robust proof of threading is that
    supplying the version SUPPRESSES the "Inferred xy_fidelity_version" warning that absence emits.
    """
    import warnings

    home = _home_from_events()
    base = {"events": _360 / "events.json", "freeze_frames": _360 / "frames.json"}
    md = _write(
        tmp_path,
        "m.json",
        {"home_team": {"home_team_id": home}, "xy_fidelity_version": "2", "shot_fidelity_version": "2"},
    )

    with warnings.catch_warnings(record=True) as explicit:
        warnings.simplefilter("always")
        L.build_statsbomb_match({**base, "metadata": md}, _MATCH_ID)
    assert not any("Inferred xy_fidelity" in str(w.message) for w in explicit), (
        "an explicit metadata fidelity must be used, not inferred"
    )

    with warnings.catch_warnings(record=True) as inferred:
        warnings.simplefilter("always")
        L.build_statsbomb_match(base, _MATCH_ID)  # no metadata -> inference
    assert any("Inferred xy_fidelity" in str(w.message) for w in inferred), (
        "absent fidelity must fall back to inference (warns)"
    )


def test_build_statsbomb_match_attaches_roster_identity(tmp_path):
    events = _events()
    pids = {e["player"]["id"] for e in events if e.get("player")}
    team = _home_from_events()
    lineup = [
        {"player_id": pid, "player_name": f"P{pid}", "jersey_number": 1, "positions": [{"position": "Midfield"}]}
        for pid in pids
    ]
    roster = _write(tmp_path, "roster.json", [{"team_id": team, "lineup": lineup}])
    paths = {"events": _360 / "events.json", "freeze_frames": _360 / "frames.json", "roster": roster}

    actions, *_ = L.build_statsbomb_match(paths, _MATCH_ID)
    assert "player_name" in actions.columns
    assert actions["player_name"].notna().any()


def test_load_matches_public_arity_unchanged(monkeypatch):
    monkeypatch.setattr(L, "_list_matches", lambda provider, tok, base: [{"id": "m1", "artifacts": {}}])
    fake = (pd.DataFrame({"action_id": [0]}), pd.DataFrame({"frame_id": [0]}), 7, None, None)
    monkeypatch.setattr(L, "_build_match_with_retry", lambda *a, **k: fake)

    out = list(L.load_matches(providers=["statsbomb"], token="x"))
    assert len(out) == 1
    assert len(out[0]) == 5  # (provider, match_id, actions, frames, home)


def test_load_statsbomb_matches_yields_visible_area(monkeypatch):
    monkeypatch.setattr(L, "_list_matches", lambda provider, tok, base: [{"id": "m1", "artifacts": {}}])
    va = pd.DataFrame({"action_id": [0], "polygon": [None]})
    fake = (pd.DataFrame({"action_id": [0]}), pd.DataFrame({"frame_id": [0]}), 7, va, None)
    monkeypatch.setattr(L, "_build_match_with_retry", lambda *a, **k: fake)

    out = list(L.load_statsbomb_matches(token="x"))
    assert len(out) == 1
    assert len(out[0]) == 6  # (provider, match_id, actions, frames, home, visible_area)
    assert out[0][5] is va
