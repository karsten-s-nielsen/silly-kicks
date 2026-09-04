"""Owner/network-gated e2e: territorial dominance runs end-to-end on real StatsBomb open data.

Product-appropriate triangulation on public data via statsbombpy (the xT-e2e substrate). Loads a few
FIFA World Cup 2022 open matches, fits ``ExpectedThreat(method="singh_counts")`` (the deterministic
classic grid) on all-but-one match, and runs ``compute_territorial_dominance`` on the HELD-OUT match
(scored match EXCLUDED from the fit -- PLAN-07). Actions stay in the raw per-acting-team-LTR frame
(``convert_to_actions`` output, ADR-028) -- NO ``play_left_to_right`` -- because the metric relates the
defender's frame to the opponent's by a 180 degree reflection. Marked e2e: deselected in the normal
suite (network + slow); skips cleanly if statsbombpy / the network is unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import statsbomb
from silly_kicks.territory import TERRITORY_COLUMNS, compute_territorial_dominance
from silly_kicks.xthreat import ExpectedThreat

_COMPETITION_ID = 43  # FIFA World Cup 2022
_SEASON_ID = 106


@pytest.mark.e2e
def test_territorial_dominance_runs_on_statsbomb_open():
    pytest.importorskip("statsbombpy")
    from statsbombpy import sb  # type: ignore[import-not-found]

    from scripts._sb_raw import flatten_events

    try:
        matches = sb.matches(competition_id=_COMPETITION_ID, season_id=_SEASON_ID, fmt="dict")
    except Exception as exc:  # network / availability
        pytest.skip(f"StatsBomb open-data unavailable: {exc}")
    if not matches:
        pytest.skip("no matches returned for the competition")

    per_match: dict[int, pd.DataFrame] = {}
    for match_id, m in list(matches.items())[:6]:  # a handful is enough to fit a serviceable grid
        try:
            home_team_id = int(m["home_team"]["home_team_id"])
            events = list(sb.events(match_id=int(match_id), fmt="dict").values())
            actions, _ = statsbomb.convert_to_actions(flatten_events(events, int(match_id)), home_team_id=home_team_id)
            per_match[int(match_id)] = actions  # RAW per-acting-team-LTR (no play_left_to_right)
        except Exception as exc:  # one bad live match must not sink the e2e
            print(f"skip {match_id}: {exc!r}")
    if len(per_match) < 3:
        pytest.skip(f"only {len(per_match)} matches converted; too few to fit + hold out")

    scored_id = next(iter(per_match))
    fit_actions = pd.concat([a for mid, a in per_match.items() if mid != scored_id], ignore_index=True)
    xt = ExpectedThreat(method="singh_counts").fit(fit_actions)  # scored match EXCLUDED from the fit

    samples, report = compute_territorial_dominance(per_match[scored_id], xt=xt)

    # schema conforms
    assert list(samples.columns) == list(TERRITORY_COLUMNS)
    for c, t in TERRITORY_COLUMNS.items():
        assert str(samples[c].dtype) == t, f"{c}: {samples[c].dtype} != {t}"
    # census conserves (ADR-042)
    assert report.n_scored + report.n_degenerate_hull + report.n_no_actions == report.n_players_in
    assert report.n_players_in > 0 and report.n_scored > 0
    # values finite + plausible on the SCORED (resolved) rows
    scored = samples[samples["territory_hull_source"] == "resolved"]
    assert len(scored) == report.n_scored
    for c in ("territory_xt_conceded", "territory_xt_prevented", "territory_hull_area_m2"):
        v = pd.to_numeric(scored[c])
        assert np.isfinite(v).all()
        assert (v >= 0).all()  # xT sums + area are non-negative
    # hull areas are within a sane pitch-scale band (a defender's third, not the whole pitch)
    assert pd.to_numeric(scored["territory_hull_area_m2"]).max() <= 105.0 * 68.0
