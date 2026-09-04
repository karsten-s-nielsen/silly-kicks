"""Native (sportec) duel path end-to-end on REAL data -- the committed IDSSE slice.

Exercises the NATIVE winner/loser extraction (sportec ``tackle_winner_*`` / ``tackle_loser_*``, ADR-001)
through the real parse-port -> sportec converter -> ``compute_duel_ratings`` chain, complementing the
derived-path StatsBomb e2e (``test_e2e.py``). The IDSSE slice is committed to the repo (real DFL match
J03WMX, reduced), so this is a REGULAR test (not ``@e2e``) that runs on every CI leg -- a stronger
guarantee than a pining-gated skip. The parse-port is pure stdlib (already CI-exercised by
``tests/providers/sportec/test_parse_port_parity.py``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from silly_kicks.duels import DUEL_COLUMNS, compute_duel_ratings, extract_duels
from silly_kicks.providers.sportec import parse_dfl_events, parse_dfl_match_info, shape_events_to_native
from silly_kicks.spadl import sportec

_FIX = Path(__file__).resolve().parents[1] / "datasets" / "sportec" / "idsse_slice"
_MATCH_ID = "J03WMX"


def _real_native_actions() -> pd.DataFrame:
    """Real sportec SPADL actions from the committed IDSSE slice (carries native tackle winner/loser)."""
    mi = parse_dfl_match_info(str(_FIX / "info.xml"))
    bronze = parse_dfl_events(str(_FIX / "events.xml"), match_info=mi, match_id=_MATCH_ID)
    native = shape_events_to_native(bronze)
    # Duels ignore geometry; home_team_start_left is required by the converter but irrelevant here.
    actions, _ = sportec.convert_to_actions(native, home_team_id=mi.home_team_id, home_team_start_left=True)
    return actions


def test_native_duels_extracted_on_real_sportec():
    actions = _real_native_actions()
    assert "tackle_winner_player_id" in actions.columns
    assert int(actions["tackle_winner_player_id"].notna().sum()) > 0  # the slice carries native winners
    games, report = extract_duels(actions)
    assert report.labeling_strategy == "native"  # native path chosen at frame-set granularity
    assert report.n_native == len(games) > 0
    assert report.n_native + report.n_derived + report.n_excluded == report.n_candidate


def test_compute_duel_ratings_on_real_sportec():
    actions = _real_native_actions()
    samples, report = compute_duel_ratings(actions)

    assert list(samples.columns) == list(DUEL_COLUMNS)
    for c, t in DUEL_COLUMNS.items():
        assert str(samples[c].dtype) == t, f"{c}: {samples[c].dtype} != {t}"
    assert report.labeling_strategy == "native"
    assert report.n_duels > 0
    assert set(samples["duel_winner_source"]) == {"native"}
    # every native duel has exactly one winner + one loser
    assert int(samples["duels_won"].sum()) == report.n_duels
    assert int(samples["duels_lost"].sum()) == report.n_duels
    # ratings actually moved off the 1500 seed (winners up, losers down)
    assert (pd.to_numeric(samples["duel_rating"]) != 1500.0).any()
    assert pd.to_numeric(samples["duel_rating"]).std() > 0.0
    # final_ratings is populated + canonical-keyed (the resume seed)
    assert len(report.final_ratings) == len(samples)
