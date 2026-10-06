"""TF-58 Task 17: degradation source + surrogate tokens, and report conservation."""

from __future__ import annotations

import dataclasses

import pytest

from silly_kicks.coordination._catalog import PairSpec
from silly_kicks.coordination._compute import (
    compute_cluster_phase,
    compute_cross_correlation,
    compute_relative_phase,
    compute_team_coordination,
    compute_vector_coding,
)
from silly_kicks.coordination._config import CoordinationParams
from silly_kicks.coordination._signals import build_coordination_signals
from silly_kicks.coordination._windows import period_windows
from tests.coordination._fixtures import make_coordination_actions, make_coordination_match

pytestmark = [
    pytest.mark.filterwarnings("ignore:vx/vy columns not found"),
    pytest.mark.filterwarnings("ignore::silly_kicks.coordination.CoordinationCoverageWarning"),
]
_FAST = dataclasses.replace(CoordinationParams(), n_surrogates=5, welch_segment_s=20.0)


class _NoGoal:
    def get(self, *_a):  # duck-typed GoalMap: every defended end unresolved
        return None


def _sig(f, params=_FAST, windows=None, **kw):
    return build_coordination_signals(
        f, windows=windows if windows is not None else period_windows(f), params=params, **kw
    )


def test_every_source_token_reachable():
    seen: set[str] = set()
    f = make_coordination_match(seconds=300.0, hz=10.0, provider="sportec")
    sig = _sig(f)
    pair, _p, _r = compute_relative_phase(sig)
    seen |= set(pair["coord_rp_source"].dropna())  # scored
    # degenerate_constant: a team variable with zero in-window variance (an immobile shape). Filtering always
    # injects ~1e-13 variance into fixture-derived signals, so inject an exactly-constant signal to hit the branch.
    degen = _sig(f)
    for tm in degen.periods[0].team_ids:
        degen.periods[0].team_signal[(tm, "spread")][:] = 5.0
    xc, _r = compute_cross_correlation(degen, pairs=[PairSpec("team_team", "spread", "spread", "canonical")])
    seen |= set(xc["coord_xc_source"].dropna())  # degenerate_constant (zero-variance signal)
    vc, _p, _r = compute_vector_coding(sig, pairs=[PairSpec("team_team", "convex_hull_area", "spread", "canonical")])
    seen |= set(vc["coord_vc_source"].dropna())  # not_commensurate (m^2 vs m)
    rp2, _p, _r = compute_relative_phase(
        sig,
        levels=["cross_variable"],
        pairs=[PairSpec("cross_variable", "centroid_x", "defensive_line_x", "attacking_defending")],
    )
    seen |= set(rp2["coord_rp_source"].dropna())  # no_possession_role (period windows only)
    tiny = _sig(f, windows=period_windows(f, length_s=0.2, step_s=0.2))
    pt, _p, _r = compute_relative_phase(tiny)
    seen |= set(pt["coord_rp_source"].dropna())  # too_short (2-sample windows)
    ct, _cp, _ts, _r = compute_cluster_phase(tiny)
    seen |= set(ct["coord_cluster_sampen_source"].dropna())  # entropy_undefined: SampEn's own column (A-20)
    sk = make_coordination_match(seconds=300.0, hz=10.0, provider="skillcorner", visibility_drop=0.95)
    psk, _p, _r = compute_relative_phase(_sig(sk))
    seen |= set(psk["coord_rp_source"].dropna())  # insufficient_detection
    small = make_coordination_match(seconds=300.0, hz=10.0, provider="sportec", n_outfield=3, with_gk=False)
    cts, _cp, _ts, _r = compute_cluster_phase(_sig(small))
    seen |= set(cts["coord_cluster_source"].dropna())  # insufficient_players (3 < min_players 6)
    gu, _p, _r = compute_relative_phase(_sig(f, goal_map=_NoGoal()))
    seen |= set(gu["coord_rp_source"].dropna())  # goal_end_unresolved

    from silly_kicks.coordination._columns import COORD_SOURCE_VALUES

    missing = set(COORD_SOURCE_VALUES) - seen
    assert not missing, missing


def test_every_surrogate_token_reachable():
    seen: set[str] = set()
    f = make_coordination_match(seconds=300.0, hz=10.0, provider="sportec")
    pair, _p, _r = compute_relative_phase(_sig(f))
    seen |= set(pair["coord_rp_surrogate_source"].dropna())  # computed + not_scored (dropped rows)
    disabled = dataclasses.replace(CoordinationParams(), n_surrogates=0)
    pd0, _p, _r = compute_relative_phase(_sig(f, params=disabled))
    seen |= set(pd0["coord_rp_surrogate_source"].dropna())  # disabled
    short = make_coordination_match(seconds=60.0, hz=10.0, provider="sportec")  # segment < 2*tau+1 (tau=600)
    pss, _p, _r = compute_relative_phase(_sig(short))
    seen |= set(pss["coord_rp_surrogate_source"].dropna())  # segment_too_short
    iaaft = dataclasses.replace(CoordinationParams(), n_surrogates=5, surrogate_method="iaaft", iaaft_max_iter=1)
    pia, _p, _r = compute_relative_phase(_sig(f, params=iaaft))
    seen |= set(pia["coord_rp_surrogate_source"].dropna())  # computed_nonconverged (max_iter=1)

    from silly_kicks.coordination._columns import COORD_SURROGATE_SOURCE_VALUES

    missing = set(COORD_SURROGATE_SOURCE_VALUES) - seen
    assert not missing, missing


def test_surrogate_methods_map_covers_every_null_family_and_matches_the_global_values():
    # A-17 anti-rot: ONE declared {family -> supported surrogate methods} map. Its union is exactly the globally
    # accepted surrogate_method values, every surrogate-bearing family is a key, and only the pair families offer iaaft.
    from silly_kicks.coordination._compute import SURROGATE_METHODS_BY_FAMILY

    assert set().union(*SURROGATE_METHODS_BY_FAMILY.values()) == {"time_shift", "iaaft"}
    assert set(SURROGATE_METHODS_BY_FAMILY) == {
        "relative_phase",
        "cross_correlation",
        "vector_coding",
        "coherence",
        "cluster",
        "team_sync",
    }
    assert {f for f, m in SURROGATE_METHODS_BY_FAMILY.items() if "iaaft" in m} == {
        "relative_phase",
        "cross_correlation",
        "vector_coding",
        "coherence",
    }


def test_cluster_and_team_sync_refuse_iaaft_but_time_shift_computes():
    # A-17: the cluster + team-sync nulls do not implement iaaft -- they must RAISE (fail loud), never silently
    # time-shift and report `computed`. time_shift still computes a real null; the pair family still honours iaaft.
    f = make_coordination_match(seconds=300.0, hz=10.0, provider="sportec", oscillation_cpm=0.5, phase_offset_deg=40.0)
    iaaft = dataclasses.replace(CoordinationParams(), n_surrogates=5, surrogate_method="iaaft", iaaft_max_iter=50)
    with pytest.raises(NotImplementedError, match=r"iaaft.*cluster"):
        compute_cluster_phase(_sig(f, params=iaaft))
    shift = _sig(f, params=dataclasses.replace(CoordinationParams(), n_surrogates=5))
    ct, _cp, ts, _r = compute_cluster_phase(shift)  # time_shift: a real null is computed here
    assert (ct["coord_cluster_surrogate_source"] == "computed").any()
    assert (ts["coord_team_sync_surrogate_source"] == "computed").any()
    pair, _p, _r = compute_relative_phase(_sig(f, params=iaaft))  # the pair family still honours iaaft
    assert pair["coord_rp_surrogate_source"].isin(["computed", "computed_nonconverged"]).any()


def test_iaaft_counts_nonconvergence_and_more_iters_help():
    # both surrogate tokens coexist at high iter (some segments converge, some do not); raising the
    # iteration budget strictly reduces the nonconvergence count -> the counter tracks real work.
    f = make_coordination_match(seconds=300.0, hz=10.0, provider="sportec")
    hard = dataclasses.replace(CoordinationParams(), n_surrogates=5, surrogate_method="iaaft", iaaft_max_iter=1)
    _pair, _p, rep_hard = compute_relative_phase(_sig(f, params=hard))
    easy = dataclasses.replace(CoordinationParams(), n_surrogates=5, surrogate_method="iaaft", iaaft_max_iter=200)
    _pair2, _p2, rep_easy = compute_relative_phase(_sig(f, params=easy))
    assert rep_hard.iaaft_nonconverged > 0
    assert rep_easy.iaaft_nonconverged < rep_hard.iaaft_nonconverged
    surr = rep_easy.surrogate_rows_by_source.get("coordination_pair", {})
    assert surr.get("computed", 0) > 0 and surr.get("computed_nonconverged", 0) > 0


def test_report_conserves_windows_and_rows():
    f = make_coordination_match(seconds=180.0, hz=10.0, provider="sportec")
    a = make_coordination_actions(f)
    res = compute_team_coordination(f, actions=a, params=_FAST)
    assert res.report.conservation_errors() == []


def test_drop_reasons_attributed_by_precedence():
    # a period window with only attacking_defending pairs -> the window's drop token is no_possession_role
    f = make_coordination_match(seconds=120.0, hz=10.0, provider="sportec")
    sig = _sig(f)
    _rp, _p, rep = compute_relative_phase(
        sig,
        levels=["cross_variable"],
        pairs=[PairSpec("cross_variable", "centroid_x", "defensive_line_x", "attacking_defending")],
    )
    assert "no_possession_role" in rep.windows_dropped
