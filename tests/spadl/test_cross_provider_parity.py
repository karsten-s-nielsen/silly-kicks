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


def _load_pff_fixture():
    from silly_kicks.spadl import pff

    # Reuse the synthetic-match loader from test_pff.py to avoid duplication.
    from tests.spadl.test_pff import _load_synthetic_events

    events = _load_synthetic_events()
    actions, _ = pff.convert_to_actions(
        events,
        home_team_id=100,
        home_team_start_left=True,
        home_team_start_left_extratime=True,
    )
    return actions


_PROVIDER_LOADERS = {
    "sportec": _load_idsse_fixture,
    "metrica": _load_metrica_fixture,
    "statsbomb": _load_statsbomb_fixture,
    "opta": _load_opta_fixture,
    "wyscout": _load_wyscout_fixture,
    "pff": _load_pff_fixture,
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
    """Sanity check: every converter returns a DataFrame with type_id present.
    Sportec output uses SPORTEC_SPADL_COLUMNS (ADR-001); other providers use
    KLOPPY_SPADL_COLUMNS or SPADL_COLUMNS.
    """
    actions = _PROVIDER_LOADERS[provider]()
    assert "type_id" in actions.columns
    assert len(actions) > 0

    if provider in ("sportec", "pff"):
        # Sportec (2.0.0) and PFF (2.6.0) output includes the 4 tackle_*_*_id
        # columns per ADR-001 (different dtypes — sportec object, PFF Int64 —
        # but same column set).
        for col in (
            "tackle_winner_player_id",
            "tackle_winner_team_id",
            "tackle_loser_player_id",
            "tackle_loser_team_id",
        ):
            assert col in actions.columns, f"{provider} output missing ADR-001 column {col!r}"


# ---------------------------------------------------------------------------
# ADR-001: caller's team_id mirrors input — never overridden from qualifiers
# (2.0.0; locks the converter contract per-provider as a regression gate)
# ---------------------------------------------------------------------------


def _input_team_values_for(provider: str) -> set[str]:
    """Capture the unique team values from each provider's input fixture.

    Returns the set of team identifiers the caller passed in. ADR-001 says
    output team_id values must be a subset of this set.
    """
    if provider == "sportec":
        parquet_path = _REPO_ROOT / "tests" / "datasets" / "idsse" / "sample_match.parquet"
        events = pd.read_parquet(parquet_path)
        return set(events["team"].dropna().astype(str).tolist())
    if provider == "metrica":
        parquet_path = _REPO_ROOT / "tests" / "datasets" / "metrica" / "sample_match.parquet"
        events = pd.read_parquet(parquet_path)
        return set(events["team"].dropna().astype(str).tolist())
    if provider == "statsbomb":
        fixture_path = _REPO_ROOT / "tests" / "datasets" / "statsbomb" / "raw" / "events" / "7298.json"
        with open(fixture_path, encoding="utf-8") as f:
            events_raw = json.load(f)
        return {str((e.get("team") or {}).get("id")) for e in events_raw if (e.get("team") or {}).get("id") is not None}
    if provider == "opta":
        # Synthetic fixture (see _load_opta_fixture); team_ids are 100, 200.
        return {"100", "200"}
    if provider == "wyscout":
        # Synthetic fixture (see _load_wyscout_fixture); team_id is 100.
        return {"100"}
    if provider == "pff":
        # Synthetic fixture (see _load_pff_fixture); two teams 100 and 200.
        return {"100", "200"}
    raise ValueError(f"unknown provider {provider!r}")


@pytest.mark.parametrize("provider", list(_PROVIDER_LOADERS.keys()))
def test_team_id_mirrors_input_team(provider: str):
    """ADR-001 contract: every converter's output team_id values must be a
    subset of the input team values. Locks the no-override contract
    per-provider as a regression gate going forward.

    Pre-2.0.0 sportec failed this gate: tackle rows had raw DFL CLU ids
    (e.g., 'DFL-CLU-000005') in team_id when the caller passed 'home' /
    'away' in the input team column.
    """
    actions = _PROVIDER_LOADERS[provider]()
    expected = _input_team_values_for(provider)

    # Cast output team_id to str for comparison (some providers use int,
    # but membership comparison via str is robust across dtypes).
    actual = set(actions["team_id"].dropna().astype(str).tolist())

    leaked = actual - expected
    assert not leaked, (
        f"Provider {provider!r} emitted team_id values not present in input: {sorted(leaked)}. "
        f"Expected subset of input team values: {sorted(expected)[:10]}{'...' if len(expected) > 10 else ''}. "
        f"This is the ADR-001 violation pattern (cf. silly-kicks pre-2.0.0 sportec tackle override)."
    )


# ---------------------------------------------------------------------------
# ADR-001 e2e on the IDSSE production fixture
# (Verifies the contract works on production-shape data — 308 rows from
# soccer_analytics.bronze.idsse_events match J03WMX, vendored 1.10.0)
# ---------------------------------------------------------------------------


class TestSportecAdrContractOnProductionFixture:
    """ADR-001 verification on production-shape IDSSE data."""

    @staticmethod
    def _load_actions():
        from silly_kicks.spadl import sportec

        parquet_path = _REPO_ROOT / "tests" / "datasets" / "idsse" / "sample_match.parquet"
        events = pd.read_parquet(parquet_path)

        # Pretend the caller normalized team to home/away labels (the
        # luxury-lakehouse adapter pattern) — overwrite events["team"]
        # using the first non-null team value as "home" and others as "away".
        first_team = events["team"].dropna().iloc[0]
        events["team"] = events["team"].apply(lambda t: "home" if t == first_team else ("away" if pd.notna(t) else t))

        actions, _ = sportec.convert_to_actions(events, home_team_id="home")
        return actions, events

    def test_no_dfl_clu_strings_leak_into_team_id(self):
        """The PR-LL2 bug: caller passed 'home'/'away' but team_id rows had
        raw 'DFL-CLU-...' strings. ADR-001 makes this impossible."""
        actions, _ = self._load_actions()
        leaked = actions["team_id"].dropna().astype(str)
        dfl_leakage = leaked[leaked.str.startswith("DFL-CLU-")]
        assert len(dfl_leakage) == 0, (
            f"DFL-CLU-... strings leaked into team_id ({len(dfl_leakage)} rows). "
            f"This is the ADR-001 violation pattern. Sample leaks: {dfl_leakage.head(3).tolist()}"
        )

    def test_team_id_only_contains_home_or_away(self):
        actions, _ = self._load_actions()
        unique_team_ids = set(actions["team_id"].dropna().astype(str).tolist())
        # The caller normalized to {home, away}. Synthetic dribble rows
        # inherit prior action's team_id (which is also home/away).
        assert unique_team_ids <= {"home", "away"}, (
            f"team_id contains values outside {{home, away}}: {sorted(unique_team_ids)}"
        )

    def test_tackle_winner_team_id_populated_for_qualifier_rows(self):
        """Some tackle rows in the fixture have tackle_winner_team qualifier
        populated (DFL-CLU-... values). The new column surfaces those verbatim."""
        actions, events = self._load_actions()

        if "tackle_winner_team" not in events.columns:
            pytest.skip("IDSSE fixture lacks tackle_winner_team column entirely")
        input_winner_rows = events[events["tackle_winner_team"].notna()]
        if len(input_winner_rows) == 0:
            pytest.skip("IDSSE fixture has no tackle rows with tackle_winner_team qualifier")

        output_winner_rows = actions[actions["tackle_winner_team_id"].notna()]
        assert len(output_winner_rows) > 0, (
            "Sportec output has zero rows with tackle_winner_team_id populated, "
            "but input has rows with the tackle_winner_team qualifier."
        )

    def test_use_tackle_winner_as_actor_round_trips(self):
        """The migration helper restores pre-2.0.0 'actor = winner' semantic."""
        from silly_kicks.spadl import use_tackle_winner_as_actor

        actions, _ = self._load_actions()

        n_winner_rows = int(actions["tackle_winner_team_id"].notna().sum())
        if n_winner_rows == 0:
            pytest.skip("IDSSE fixture has no rows with tackle_winner_team_id; helper has nothing to swap")

        rotated = use_tackle_winner_as_actor(actions)

        # On rows with non-null winner cols, team_id is now the winner team
        # id (not 'home'/'away').
        rotated_winner_rows = rotated[actions["tackle_winner_team_id"].notna()]
        assert (rotated_winner_rows["team_id"] == rotated_winner_rows["tackle_winner_team_id"]).all(), (
            "use_tackle_winner_as_actor failed to overwrite team_id on rows with non-null winner cols"
        )

    def test_keeper_coverage_preserved_from_1_10_0(self):
        """ADR-001 changes don't regress the 1.10.0 keeper coverage fix."""
        from silly_kicks.spadl import coverage_metrics

        actions, _ = self._load_actions()
        m = coverage_metrics(
            actions=actions,
            expected_action_types={
                "keeper_save",
                "keeper_claim",
                "keeper_punch",
                "keeper_pick_up",
            },
        )
        keeper_total = sum(
            m["counts"].get(t, 0) for t in ("keeper_save", "keeper_claim", "keeper_punch", "keeper_pick_up")
        )
        # IDSSE fixture has 7 throwOut + 1 punt qualifier rows from PR-S10;
        # each produces a keeper_pick_up. Plus possibly save/claim/punch rows.
        # Assert at least 1 (the 1.10.0 floor); typical is ~8.
        assert keeper_total > 0, f"1.10.0 keeper coverage regressed under ADR-001 changes. counts={m['counts']}"
