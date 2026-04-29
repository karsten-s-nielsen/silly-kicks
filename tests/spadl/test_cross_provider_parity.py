"""Cross-provider parity meta-test (added in silly-kicks 1.10.0).

For every DataFrame SPADL converter, asserts that, when given a fixture
that exercises GK paths (with ``goalkeeper_ids`` where the source format
requires it), the output contains at least one ``keeper_*`` action.

Pre-1.10.0 this test would have failed for sportec (Pass→Play bug + missing
throwOut/punt vocabulary) and metrica (no native GK markers + no
goalkeeper_ids parameter). Post-1.10.0 all 5 DataFrame converters pass —
ensuring future converter regressions in the keeper-action emission class
surface immediately.

The Wyscout DataFrame converter is exercised via a synthetic fixture (an
aerial duel by a known goalkeeper, since vendoring a Wyscout production
fixture is out of scope for PR-S10).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.spadl import coverage_metrics

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_KEEPER_TYPE_NAMES = frozenset({"keeper_save", "keeper_claim", "keeper_punch", "keeper_pick_up"})


# ---------------------------------------------------------------------------
# Adapters: load each provider's vendored fixture and run convert_to_actions
# ---------------------------------------------------------------------------


def _load_idsse_fixture():
    from silly_kicks.spadl import sportec

    parquet_path = _REPO_ROOT / "tests" / "datasets" / "idsse" / "sample_match.parquet"
    if not parquet_path.exists():
        pytest.fail(
            f"IDSSE fixture not found at {parquet_path}. Regenerate via "
            f"scripts/extract_provider_fixtures.py --provider idsse."
        )
    events = pd.read_parquet(parquet_path)
    home_team = events["team"].dropna().iloc[0]
    # GK player_ids recoverable from rows where play_goal_keeper_action
    # is non-null in the fixture itself.
    gk_ids: set[str] | None = None
    if "play_goal_keeper_action" in events.columns:
        gk_ids = set(events.loc[events["play_goal_keeper_action"].notna(), "player_id"].dropna().astype(str).tolist())
    actions, _ = sportec.convert_to_actions(events, home_team_id=str(home_team), goalkeeper_ids=gk_ids)
    return actions


def _load_metrica_fixture():
    from silly_kicks.spadl import metrica

    parquet_path = _REPO_ROOT / "tests" / "datasets" / "metrica" / "sample_match.parquet"
    if not parquet_path.exists():
        pytest.fail(
            f"Metrica fixture not found at {parquet_path}. Regenerate via "
            f"scripts/extract_provider_fixtures.py --provider metrica."
        )
    events = pd.read_parquet(parquet_path)
    home_team = events["team"].dropna().iloc[0]
    # For Metrica we can't derive GK ids from the source format (no GK
    # markers exist). Use the first PASS player (with a non-null player_id)
    # from the home team as the "assumed GK" to surface at least one synth
    # path. Real callers supply goalkeeper_ids from squad metadata.
    home_passes = events[(events["type"] == "PASS") & (events["team"] == home_team) & events["player"].notna()]
    if home_passes.empty:
        pytest.skip("Metrica fixture lacks any PASS-by-home-team event with a known player_id")
    gk_id = str(home_passes["player"].iloc[0])
    actions, _ = metrica.convert_to_actions(events, home_team_id=str(home_team), goalkeeper_ids={gk_id})
    return actions


def _load_statsbomb_fixture():
    from silly_kicks.spadl import statsbomb

    fixture_path = _REPO_ROOT / "tests" / "datasets" / "statsbomb" / "raw" / "events" / "7298.json"
    if not fixture_path.exists():
        pytest.fail(
            f"StatsBomb fixture not found at {fixture_path}. See "
            f"tests/datasets/statsbomb/README.md for vendoring details."
        )

    with open(fixture_path, encoding="utf-8") as f:
        events_raw = json.load(f)

    _top_level_keys = {"id", "period", "timestamp", "team", "player", "type", "location"}
    adapted = pd.DataFrame(
        [
            {
                "game_id": 7298,
                "event_id": e.get("id"),
                "period_id": e.get("period"),
                "timestamp": e.get("timestamp"),
                "team_id": (e.get("team") or {}).get("id"),
                "player_id": (e.get("player") or {}).get("id"),
                "type_name": (e.get("type") or {}).get("name"),
                "location": e.get("location"),
                "extra": {k: v for k, v in e.items() if k not in _top_level_keys},
            }
            for e in events_raw
        ]
    )
    home_team_id = int(adapted["team_id"].dropna().iloc[0])
    actions, _ = statsbomb.convert_to_actions(adapted, home_team_id=home_team_id)
    return actions


def _load_opta_fixture():
    from silly_kicks.spadl import opta

    # Synthetic Opta-shape fixture exercising GK actions (no production
    # Opta sample is currently vendored; this synthetic one verifies the
    # converter emits keeper_save when given an appropriate "save" event).
    events = pd.DataFrame(
        {
            "game_id": [1, 1],
            "event_id": ["e1", "e2"],
            "period_id": [1, 1],
            "minute": [0, 0],
            "second": [1, 2],
            "team_id": [100, 200],
            "player_id": [201, 100],
            "type_name": ["pass", "save"],
            "outcome": [True, True],
            "start_x": [50.0, 5.0],
            "start_y": [50.0, 50.0],
            "end_x": [60.0, 5.0],
            "end_y": [50.0, 50.0],
            "qualifiers": [{}, {}],
        }
    )
    actions, _ = opta.convert_to_actions(events, home_team_id=100)
    return actions


def _load_wyscout_fixture():
    from silly_kicks.spadl import wyscout

    # Synthetic Wyscout-shape fixture exercising the goalkeeper_ids
    # aerial-duel reclassification path (live since 1.0.0).
    events = pd.DataFrame(
        {
            "game_id": [1],
            "event_id": [1],
            "period_id": [1],
            "milliseconds": [1000],
            "team_id": [100],
            "player_id": [200],
            "type_id": [1],  # _WS_TYPE_DUEL
            "subtype_id": [10],  # _WS_SUBTYPE_AIR_DUEL
            "positions": [[{"x": 10, "y": 50}, {"x": 10, "y": 50}]],
            "tags": [[{"id": 703}]],  # tag 703 = "won"
        }
    )
    actions, _ = wyscout.convert_to_actions(events, home_team_id=100, goalkeeper_ids={200})
    return actions


_PROVIDER_LOADERS = {
    "sportec": _load_idsse_fixture,
    "metrica": _load_metrica_fixture,
    "statsbomb": _load_statsbomb_fixture,
    "opta": _load_opta_fixture,
    "wyscout": _load_wyscout_fixture,
}


# ---------------------------------------------------------------------------
# The parity gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", list(_PROVIDER_LOADERS.keys()))
def test_converter_emits_at_least_one_keeper_action(provider: str):
    """Each DataFrame SPADL converter, given a fixture exercising GK paths
    (with appropriate goalkeeper_ids where the source format requires it),
    must emit at least one keeper_* action.

    This test would have caught Bugs 1-3 in silly-kicks 1.7.0 if it had
    existed.
    """
    actions = _PROVIDER_LOADERS[provider]()
    m = coverage_metrics(actions=actions, expected_action_types=set(_KEEPER_TYPE_NAMES))
    keeper_count_total = sum(m["counts"].get(t, 0) for t in _KEEPER_TYPE_NAMES)
    assert keeper_count_total > 0, (
        f"Provider {provider!r} emitted zero keeper_* actions on its fixture. "
        f"Coverage breakdown: counts={m['counts']}, missing={m['missing']}."
    )


@pytest.mark.parametrize("provider", list(_PROVIDER_LOADERS.keys()))
def test_converter_returns_canonical_spadl_columns(provider: str):
    """Sanity check: every converter returns a DataFrame with type_id present."""
    actions = _PROVIDER_LOADERS[provider]()
    assert "type_id" in actions.columns
    assert len(actions) > 0
