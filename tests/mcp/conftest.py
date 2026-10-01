"""Fixtures for the Phase 2 MCP tool tests.

The geometry OK-side uses a REAL committed match slice (elastic_sync j03wmx — real tracking frames
with genuine per-period coords + labels), not a synthesized one. The mislabeled / unoriented /
untracked variants are DERIVED from it. Server tests monkeypatch ``_load.load_match`` to serve these
LoadedMatches from a ref registry (no network); ``tests/mcp`` has NO ``__init__.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from _loader_pining import LoadedMatch  # noqa: E402

_SLICE = Path(__file__).resolve().parents[1] / "datasets" / "elastic_sync" / "j03wmx_slice"


def _slice_frames_actions() -> tuple[pd.DataFrame, pd.DataFrame, object]:
    frames = pd.read_parquet(_SLICE / "frames.parquet")
    actions = pd.read_parquet(_SLICE / "actions.parquet")
    # Home = the team that attacks +x (label "ltr") in the real slice; its GK sits at low x, so the
    # bound geometry function does not reflect period 1 and the geometric direction == the stored
    # label -> OK (non-vacuously, on real frames).
    players = frames[~frames["is_ball"].astype(bool)]
    ltr = players.loc[players["team_attacking_direction"] == "ltr", "team_id"].dropna()
    home_team_id = ltr.iloc[0] if len(ltr) else players["team_id"].dropna().iloc[0]
    return actions, frames, home_team_id


def _loaded(actions: pd.DataFrame, frames: pd.DataFrame, home_team_id: object) -> LoadedMatch:
    return LoadedMatch("idsse", "j03wmx", actions, frames, home_team_id, None, None)


def _build_ok() -> LoadedMatch:
    actions, frames, home = _slice_frames_actions()
    return _loaded(actions, frames, home)


def _build_mislabeled() -> LoadedMatch:
    actions, frames, home = _slice_frames_actions()
    frames = frames.copy()
    from silly_kicks.id_compat import ids_match

    home_rows = ids_match(frames["team_id"], home).fillna(False).to_numpy(dtype=bool)
    tad = frames["team_attacking_direction"].to_numpy(dtype=object).copy()
    # Flip ONLY the home team's present labels -> they now contradict the (unchanged) coords.
    for i in range(len(tad)):
        if home_rows[i] and tad[i] in ("ltr", "rtl"):
            tad[i] = "rtl" if tad[i] == "ltr" else "ltr"
    frames["team_attacking_direction"] = tad
    return _loaded(actions, frames, home)


def _build_untracked_home_gk() -> LoadedMatch:
    """Two periods: P1 home-GK absent / away GK present (away-fallback path); P2 no GK at all
    (no-anchor -> excluded). The P2 home label is deliberately "rtl" so a BROKEN exclusion would
    read P2 as unreflected "ltr" and fire a false MISMATCH (r7 CONSIDER)."""
    actions, frames, home = _slice_frames_actions()
    from silly_kicks.id_compat import ids_match

    gk = frames["is_goalkeeper"].astype(bool).to_numpy()
    home_rows = ids_match(frames["team_id"], home).fillna(False).to_numpy(dtype=bool)
    p1 = frames[~(gk & home_rows)].copy()  # drop home GK; away GK stays -> away fallback
    p2 = frames[~gk].copy()  # no GK at all -> no-anchor period
    p2["period_id"] = 2
    p2["frame_id"] = p2["frame_id"] + int(frames["frame_id"].max()) + 1
    p2_home = ids_match(p2["team_id"], home).fillna(False).to_numpy(dtype=bool)
    tad2 = p2["team_attacking_direction"].to_numpy(dtype=object).copy()
    for i in range(len(tad2)):
        if p2_home[i]:
            tad2[i] = "rtl"
    p2["team_attacking_direction"] = tad2
    return _loaded(actions, pd.concat([p1, p2], ignore_index=True), home)


@pytest.fixture
def _registry() -> dict[str, LoadedMatch]:
    return {
        "OK": _build_ok(),
        "MIS": _build_mislabeled(),
        "UNTRACKED": _build_untracked_home_gk(),
    }


@pytest.fixture
def _server_load(monkeypatch, _registry):
    """Serve the registry LoadedMatches from ``_load.load_match`` (no network); unknown ref RAISES."""
    from silly_kicks.mcp import _load

    def fake(match_ref, provider=None):
        if match_ref in _registry:
            return _registry[match_ref]
        raise RuntimeError(f"no match resolved for {match_ref!r} (empty/tokenless)")

    monkeypatch.setattr(_load, "load_match", fake)
    return _registry


@pytest.fixture
def fixture_ref(_server_load) -> str:
    return "OK"


@pytest.fixture
def mislabeled_ref(_server_load) -> str:
    return "MIS"


@pytest.fixture
def untracked_home_gk_ref(_server_load) -> str:
    return "UNTRACKED"


@pytest.fixture
def dirty_tree(tmp_path, monkeypatch):
    """A working-tree dirtier that also asserts no new artifact appeared. The MCP tools never write,
    so this passes; it is the read-only-on-dirty-tree guard (spec §7)."""

    class _Dirty:
        def __init__(self) -> None:
            self._marker = Path("_mcp_dirty_marker.tmp")
            self._marker.write_text("dirty", encoding="utf-8")
            self._before = self._artifacts()

        @staticmethod
        def _artifacts() -> set[str]:
            return {str(p) for p in Path("docs/research").rglob("*.json")}

        def no_new_artifacts(self) -> bool:
            return self._artifacts() == self._before

        def cleanup(self) -> None:
            self._marker.unlink(missing_ok=True)

    d = _Dirty()
    yield d
    d.cleanup()
