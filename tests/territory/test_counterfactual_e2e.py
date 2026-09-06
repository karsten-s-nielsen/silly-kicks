"""Owner/network-gated e2e: the TF-54b counterfactual territorial "prevented" valuation runs
end-to-end on real StatsBomb open data.

Sibling of ``tests/territory/test_e2e.py`` (v1 ``completed_failed``) for ``method="counterfactual"``
(spec section 5, TF-54b). Loads a handful of FIFA World Cup 2022 open matches, fits
``ExpectedThreat(method="singh_counts")`` AND ``silly_kicks.expected_passing.PassCompletionModel`` on
all-but-one match -- the leakage discipline of spec section 7.1: both INJECTED models are fit on a
corpus DISJOINT from the scored match -- and runs ``compute_territorial_dominance(...,
method="counterfactual")`` on the HELD-OUT match. Actions stay in the raw per-acting-team-LTR frame
(``convert_to_actions`` output, ADR-028) -- NO ``play_left_to_right`` -- because the metric relates the
defender's frame to the opponent's frame by a 180 degree reflection, same as the v1 e2e test.

This is the SHAPE / SCHEMA / CONSERVATION smoke test on real data (owner-run only -- ``-m "not e2e"``
excludes it from CI). The pre-registered CONSTRUCT-VALIDITY battery (completion AUC / ECE / Brier,
synthetic-interception target recovery versus the naive baselines, the locked elite-defender prior,
reliability / discriminant) is a SEPARATE owner-run corpus pass --
``scripts/validate_territory_counterfactual.py`` (spec section 7) -- whose artifact is what this
cycle's ADR/CHANGELOG findings are filled in from. Marked e2e: deselected in the normal suite (network
+ slow); skips cleanly if statsbombpy / the network is unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.expected_passing import PassCompletionModel
from silly_kicks.spadl import statsbomb
from silly_kicks.territory import (
    TERRITORY_TARGET_SOURCE_VALUES,
    columns_for_method,
    compute_territorial_dominance,
)
from silly_kicks.xthreat import ExpectedThreat

_COMPETITION_ID = 43  # FIFA World Cup 2022
_SEASON_ID = 106


@pytest.mark.e2e
def test_territorial_dominance_counterfactual_runs_on_statsbomb_open():
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
    for match_id, m in list(matches.items())[:6]:  # a handful is enough to fit a serviceable xt + completion model
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
    # Both injected ports are fit on the corpus DISJOINT from the scored match (spec section 7.1
    # leakage discipline) -- the held-out match is excluded from both fits, not just the xT one.
    xt = ExpectedThreat(method="singh_counts").fit(fit_actions)
    completion_model = PassCompletionModel().fit(fit_actions)

    samples, report = compute_territorial_dominance(
        per_match[scored_id],
        xt=xt,
        method="counterfactual",
        completion_model=completion_model,
    )

    # schema conforms to the method-dependent counterfactual schema (SPEC-04) -- names AND dtypes
    expected_schema = columns_for_method("counterfactual")
    assert list(samples.columns) == list(expected_schema)
    for c, t in expected_schema.items():
        assert str(samples[c].dtype) == t, f"{c}: {samples[c].dtype} != {t}"

    # census conserves (ADR-042)
    assert report.n_scored + report.n_degenerate_hull + report.n_no_actions == report.n_players_in
    assert report.n_players_in > 0 and report.n_scored > 0
    assert report.n_target_modeled >= 0 and report.n_target_unresolved >= 0

    scored = samples[samples["territory_hull_source"] == "resolved"]
    assert len(scored) == report.n_scored

    # Every resolved row's core valuation columns are initialized 0.0 and only ever accumulate a
    # non-negative xT * completion-probability contribution (spec section 5.7), so they are ALWAYS
    # finite and non-negative -- never NaN, never fabricated-negative.
    for c in (
        "territory_xt_conceded",
        "territory_xt_prevented",
        "territory_xt_conceded_forward",
        "territory_xt_prevented_forward",
        "territory_expected_threat_faced",
        "territory_hull_area_m2",
    ):
        v = pd.to_numeric(scored[c])
        assert np.isfinite(v).all(), f"{c} not finite"
        assert (v >= 0).all(), f"{c} has a negative value"

    # xt_net / the GSAA-style expected-minus-realized headline are SIGNED by construction (a
    # defender can be "worse than expectation") but must stay finite on every resolved row.
    for c in ("territory_xt_net", "territory_xt_prevented_above_expectation"):
        v = pd.to_numeric(scored[c])
        assert np.isfinite(v).all(), f"{c} not finite"

    # counts are non-negative
    for c in ("territory_passes_into_hull", "territory_passes_aimed_into_hull"):
        v = pd.to_numeric(scored[c])
        assert (v >= 0).all()

    # mean_completion_faced is a probability -> bounded [0, 1] wherever a defender faced an aimed-in
    # pass (NaN when passes_aimed_into_hull == 0 -- never a fabricated mean of nothing), and its
    # per-defender target-source provenance is a real token from the closed vocabulary.
    faced = scored[scored["territory_passes_aimed_into_hull"] > 0]
    if len(faced):
        mc = pd.to_numeric(faced["territory_mean_completion_faced"])
        assert np.isfinite(mc).all()
        assert ((mc >= 0) & (mc <= 1)).all()
        assert faced["territory_target_source"].isin(TERRITORY_TARGET_SOURCE_VALUES).all()

    # hull areas are within a sane pitch-scale band (a defender's third, not the whole pitch)
    assert pd.to_numeric(scored["territory_hull_area_m2"]).max() <= 105.0 * 68.0
