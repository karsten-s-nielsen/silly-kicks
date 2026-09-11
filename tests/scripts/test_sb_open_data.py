"""Tests for ``scripts/_sb_open_data.py`` -- the PUBLIC StatsBomb open-data loader (WC2022 men's).

The pining ``statsbomb`` provider is a private women's corpus, so the TF-54b bundled pass-completion
model and the construct-validity battery load the public men's FIFA World Cup 2022 open data instead
(the corpus the locked elite-defender prior matches). The load-bearing new logic is the
``player_id -> player_name`` join that gives the elite-prior name resolution real surnames to match;
the pure helpers are exercised offline here, and an ``@e2e`` smoke pins the full network contract.
"""

from __future__ import annotations

import pandas as pd
import pytest

import scripts._sb_open_data as loader

# --- pure helpers (offline) ------------------------------------------------------------------


def test_player_id_to_name_builds_the_join_and_tolerates_missing():
    events = [
        {"player": {"id": 3097, "name": "Virgil van Dijk"}},
        {"player": {"id": 3097, "name": "Virgil van Dijk"}},  # dedup: same id, kept once
        {"player": {"id": 5487, "name": "Josko Gvardiol"}},
        {"player": None},  # a team/formation event with no actor -- skipped
        {"type": {"name": "Half Start"}},  # no player key at all -- skipped
        {"player": {"id": 99, "name": None}},  # id present, name missing -> None value
    ]
    got = loader._player_id_to_name(events)
    assert got == {3097: "Virgil van Dijk", 5487: "Josko Gvardiol", 99: None}


def test_values_handles_both_dict_and_list_payloads():
    # statsbombpy fmt="dict" returns id-keyed dict; older/list versions return a list.
    assert loader._values({10: "a", 20: "b"}) == ["a", "b"]
    assert loader._values(["a", "b"]) == ["a", "b"]


def test_world_cup_2022_constant_is_the_locked_corpus():
    # The bundled model + battery default to this; a drift here is a corpus change, not a tweak.
    assert loader.WORLD_CUP_2022 == (43, 106)


# --- full loader contract (network; deselected in CI like test_sb360_open_e2e) ----------------


@pytest.mark.e2e
def test_load_open_data_matches_yields_named_event_only_tuples():
    pytest.importorskip("statsbombpy")

    seen = 0
    for provider, match_id, actions, frames, home in loader.load_open_data_matches(
        competition_id=43, season_id=106, max_matches=2
    ):
        seen += 1
        assert provider == "statsbomb"
        assert isinstance(match_id, str)
        assert isinstance(home, int)
        assert frames.empty, "event-only: frames must be an empty DataFrame"
        assert "player_name" in actions.columns
        # Every acting row on a real match has a resolved surname (the elite-prior join target).
        acted = actions[actions["player_id"].notna()]
        assert len(acted) > 0
        assert acted["player_name"].notna().all()
        assert isinstance(actions, pd.DataFrame) and len(actions) > 500
    assert seen == 2
