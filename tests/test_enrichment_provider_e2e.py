"""Cross-provider e2e regression for the NaN-safety contract (ADR-003).

Runs every @nan_safe_enrichment helper against vendored production-shape
fixtures from each supported provider. Catches: helper crashes on real
production data shape from a provider whose data shape differs from the
ones used during helper development.

Production-shape fixtures used:
- StatsBomb: tests/datasets/statsbomb/spadl-WorldCup-2018.h5 (1 match)
- IDSSE (DFL Sportec via sportec converter): tests/datasets/idsse/sample_match.parquet
- Metrica (via metrica converter): tests/datasets/metrica/sample_match.parquet

For atomic helpers we currently exercise the StatsBomb-derived atomic-SPADL
fixture only — atomic conversion of IDSSE / Metrica fixtures requires
verifying the atomic converter pipeline against those providers, which is
out of scope for this PR.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

import silly_kicks.atomic.spadl.utils as atomic_utils
import silly_kicks.spadl.utils as std_utils

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _discover(module) -> tuple:
    return tuple(fn for _, fn in inspect.getmembers(module, inspect.isfunction) if getattr(fn, "_nan_safe", False))


STD_ENRICHMENTS = _discover(std_utils)
ATOMIC_ENRICHMENTS = _discover(atomic_utils)


# ---------------------------------------------------------------------------
# Provider fixture loaders — each returns a SPADL DataFrame ready for
# enrichment helpers. Adapted from tests/spadl/test_cross_provider_parity.py
# patterns.
# ---------------------------------------------------------------------------


def _load_statsbomb_one_match() -> pd.DataFrame:
    """First match from the vendored WorldCup-2018 SPADL HDF5 fixture."""
    h5_path = _REPO_ROOT / "tests" / "datasets" / "statsbomb" / "spadl-WorldCup-2018.h5"
    if not h5_path.exists():
        pytest.skip(f"StatsBomb HDF5 fixture not found at {h5_path}")
    games = pd.read_hdf(h5_path, key="games")
    if isinstance(games, pd.Series):
        games = games.to_frame()
    first_game_id = games.iloc[0]["game_id"]
    actions = pd.read_hdf(h5_path, key=f"actions/game_{first_game_id}")
    if isinstance(actions, pd.Series):
        actions = actions.to_frame()
    return actions


def _load_idsse_via_sportec() -> pd.DataFrame:
    """IDSSE bronze events → sportec converter → SPADL."""
    from silly_kicks.spadl import sportec

    parquet_path = _REPO_ROOT / "tests" / "datasets" / "idsse" / "sample_match.parquet"
    if not parquet_path.exists():
        pytest.skip(f"IDSSE fixture not found at {parquet_path}")
    events = pd.read_parquet(parquet_path)
    # IDSSE bronze fixture's ``team`` column uses literal "home"/"away" labels
    # (not team_ids). Pass "home" so the converter mirrors only away-team rows
    # per the ABSOLUTE_FRAME_HOME_RIGHT contract. PR-S22 fixed the prior
    # ``events["team"].iloc[0]`` heuristic which picked "away" because the
    # first row in the fixture happens to be by the away team.
    gk_ids: set[str] | None = None
    if "play_goal_keeper_action" in events.columns:
        gk_ids = set(events.loc[events["play_goal_keeper_action"].notna(), "player_id"].dropna().astype(str).tolist())
    actions, _ = sportec.convert_to_actions(
        events, home_team_id="home", goalkeeper_ids=gk_ids, home_team_start_left=True
    )
    return actions


def _load_metrica_via_metrica() -> pd.DataFrame:
    """Metrica events → metrica converter → SPADL."""
    from silly_kicks.spadl import metrica

    parquet_path = _REPO_ROOT / "tests" / "datasets" / "metrica" / "sample_match.parquet"
    if not parquet_path.exists():
        pytest.skip(f"Metrica fixture not found at {parquet_path}")
    events = pd.read_parquet(parquet_path)
    home_team = events["team"].dropna().iloc[0]
    home_passes = events[(events["type"] == "PASS") & (events["team"] == home_team) & events["player"].notna()]
    if home_passes.empty:
        pytest.skip("Metrica fixture lacks any PASS-by-home-team event with a known player_id")
    gk_id = str(home_passes["player"].iloc[0])
    actions, _ = metrica.convert_to_actions(
        events, home_team_id=str(home_team), goalkeeper_ids={gk_id}, home_team_start_left=True
    )
    return actions


@pytest.fixture(params=["statsbomb", "idsse", "metrica"])
def std_provider_actions(request) -> pd.DataFrame:
    """SPADL DataFrame from one production-shape provider fixture (parametrized)."""
    if request.param == "statsbomb":
        return _load_statsbomb_one_match()
    if request.param == "idsse":
        return _load_idsse_via_sportec()
    return _load_metrica_via_metrica()


# ---------------------------------------------------------------------------
# Auto-discovered cross-provider e2e: every decorated helper x every provider.
# Failure mode: helper crashes on real production data shape from any provider.
# ---------------------------------------------------------------------------


def test_provider_registry_nonempty() -> None:
    """Bulletproof: at least 5 standard helpers + 5 atomic helpers in registry."""
    assert len(STD_ENRICHMENTS) >= 5, [fn.__name__ for fn in STD_ENRICHMENTS]
    assert len(ATOMIC_ENRICHMENTS) >= 5, [fn.__name__ for fn in ATOMIC_ENRICHMENTS]


@pytest.mark.parametrize("helper", STD_ENRICHMENTS, ids=lambda h: h.__name__)
def test_standard_helper_provider_e2e(helper, std_provider_actions) -> None:
    """Every @nan_safe_enrichment standard helper produces a DataFrame
    on production-shape input from each supported provider.
    """
    out = helper(std_provider_actions)
    assert isinstance(out, pd.DataFrame), f"{helper.__name__} returned {type(out).__name__} on provider data"
    assert len(out) == len(std_provider_actions), (
        f"{helper.__name__} changed row count on provider data ({len(std_provider_actions)} -> {len(out)})"
    )


@pytest.fixture
def atomic_statsbomb_actions() -> pd.DataFrame:
    """Atomic-SPADL DataFrame from a StatsBomb one-match conversion."""
    from silly_kicks.atomic.spadl import convert_to_atomic

    std_actions = _load_statsbomb_one_match()
    return convert_to_atomic(std_actions)


@pytest.mark.parametrize("helper", ATOMIC_ENRICHMENTS, ids=lambda h: h.__name__)
def test_atomic_helper_provider_e2e(helper, atomic_statsbomb_actions) -> None:
    """Every @nan_safe_enrichment atomic helper produces a DataFrame on
    StatsBomb-derived atomic-SPADL data.
    """
    out = helper(atomic_statsbomb_actions)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(atomic_statsbomb_actions)
