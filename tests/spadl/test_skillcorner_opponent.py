"""Task 2.2 — SkillCorner opponent resolution from events (decision C6, spec section 5.2).

``_resolve_opponent`` returns the single opponent team id (the distinct team id that is not
``home_team.id``), comparing ids through ``id_compat`` so a float-typed id resolves correctly. It
raises on a home-id mismatch, an away-authority mismatch, or more than two teams, and returns
``None`` for a single-team slice (no ``_against`` restart derivable).
"""

import pandas as pd
import pytest

from silly_kicks.spadl.skillcorner import _resolve_opponent


def _pp(team_ids) -> pd.DataFrame:
    return pd.DataFrame({"team_id": team_ids})


def _meta(home_id, away_id=None, match_id="M1") -> dict:
    md: dict = {"id": match_id, "home_team": {"id": home_id}}
    if away_id is not None:
        md["away_team"] = {"id": away_id}
    return md


def test_opponent_is_the_other_distinct_team():
    assert _resolve_opponent(_pp(["100", "200", "100"]), _meta("100")) == "200"


def test_opponent_resolved_when_home_appears_as_marker_team():
    assert _resolve_opponent(_pp(["200", "100"]), _meta("100", away_id="200")) == "200"


def test_home_not_in_event_teams_raises_naming_match_id():
    with pytest.raises(ValueError, match="M1"):
        _resolve_opponent(_pp(["200", "300"]), _meta("100", match_id="M1"))


def test_away_present_but_event_set_mismatch_raises():
    # away_team.id present (200); event teams {100, 300} != {100, 200}.
    with pytest.raises(ValueError, match="M1"):
        _resolve_opponent(_pp(["100", "300"]), _meta("100", away_id="200"))


def test_away_absent_does_not_raise():
    assert _resolve_opponent(_pp(["100", "200"]), _meta("100")) == "200"


def test_more_than_two_teams_raises():
    with pytest.raises(ValueError, match="M1"):
        _resolve_opponent(_pp(["100", "200", "300"]), _meta("100"))


def test_single_team_returns_none_unresolved():
    assert _resolve_opponent(_pp(["100", "100"]), _meta("100")) is None


def test_float_team_id_resolves_via_id_compat_not_str():
    # home id 866 (int in json); pp team ids are float 866.0 / 512.0. str() would give "866.0".
    assert _resolve_opponent(_pp([866.0, 512.0, 866.0]), _meta(866)) == "512"
