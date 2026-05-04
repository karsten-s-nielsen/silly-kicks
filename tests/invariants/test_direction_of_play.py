"""Physical invariant: every team's shots cluster at high-x in SPADL output.

PR-S22: closes the test-layer gap that let the v0.1.0 direction-of-play bug
survive 9 minor releases. SPADL convention is "all teams attack left-to-right"
-- so for any correctly-converted match, both teams' shots must average above
``field_length / 2`` in their start_x.

Parametrized across every committed provider fixture x xy_fidelity_version
axis for StatsBomb. On main (pre-PR-S22) this test fails for native StatsBomb,
native Wyscout, and synthetic Wyscout 2-team -- the proof that the test layer
was missing.

See ADR-006 for the full architectural rationale.
"""

from __future__ import annotations

import pytest

from silly_kicks.spadl import config as spadlconfig

from . import _loaders


def _statsbomb_loader_factory(match_id: int, fidelity: int):
    def loader():
        return _loaders.load_statsbomb(match_id, xy_fidelity_version=fidelity)

    return loader


_BASE_CASES = [
    # (provider_label, loader, n_min_shots_per_team)
    ("wyscout_2team_synthetic", _loaders.load_wyscout_2team_synthetic, 3),
    ("opta_2team_synthetic", _loaders.load_opta_2team_synthetic, 3),
    ("sportec_native", _loaders.load_sportec_native, 1),
    ("sportec_via_kloppy", _loaders.load_sportec_via_kloppy, 1),
    ("metrica_native", _loaders.load_metrica_native, 1),
    # Excluded: metrica_via_kloppy. The vendored kloppy metrica_events.json and
    # epts_metrica_metadata.xml are from DIFFERENT matches per
    # tests/datasets/kloppy/README.md, so team_id mapping between events and
    # metadata is not reliable; kloppy may assign HOME/AWAY incorrectly. The
    # kloppy gateway path is exercised via sportec_via_kloppy above.
    ("pff_synthetic", _loaders.load_pff_synthetic, 1),
]


_STATSBOMB_CASES = [
    (f"statsbomb_{match_id}_fid{fid}", _statsbomb_loader_factory(match_id, fid), 5)
    for match_id in (7298, 7584, 3754058)
    for fid in (1, 2)
]


_ALL_CASES = _BASE_CASES + _STATSBOMB_CASES


_SHOT_TYPE_IDS = frozenset(spadlconfig.actiontype_id[name] for name in ("shot", "shot_penalty", "shot_freekick"))


@pytest.mark.parametrize("provider,loader,n_min_shots_per_team", _ALL_CASES)
def test_per_team_shots_attack_high_x(provider: str, loader, n_min_shots_per_team: int):
    """Every team's shots must average start_x > field_length / 2 (SPADL LTR).

    Counts all SPADL shot variants (``shot`` / ``shot_penalty`` /
    ``shot_freekick``) -- the direction-of-play invariant is type-agnostic;
    a converter's set-piece-composition rules can upgrade ``SHOT`` to
    ``shot_freekick`` (Metrica) without changing the geometric invariant.

    Tolerance: PFF / Sportec / Metrica synthetic fixtures may have only one
    team with shots in the limited fixture window; we check what's available
    and skip teams below ``n_min_shots_per_team``.
    """
    actions, _home_team_id = loader()
    shots = actions[actions["type_id"].isin(_SHOT_TYPE_IDS)]
    assert len(shots) > 0, (
        f"{provider}: fixture has no shot/shot_penalty/shot_freekick actions; regression in fixture or converter."
    )

    by_team = shots.groupby("team_id").agg(n=("start_x", "size"), mean_x=("start_x", "mean"))
    reliable = by_team[by_team["n"] >= n_min_shots_per_team]
    assert not reliable.empty, (
        f"{provider}: no team has >= {n_min_shots_per_team} shots in fixture; "
        f"got {by_team.to_dict('index')}. Lower n_min for this provider or "
        "add more shot-rich rows to the fixture."
    )

    # The actual invariant: each reliable team's mean shot start_x is above field midpoint.
    failing = reliable[reliable["mean_x"] <= spadlconfig.field_length / 2]
    assert failing.empty, (
        f"{provider}: teams with mean shot x at wrong end of pitch (SPADL LTR violated): "
        f"{failing.to_dict('index')}. All teams should attack high-x. Full breakdown: "
        f"{by_team.to_dict('index')}"
    )


def test_at_least_one_provider_has_two_teams_with_shots():
    """Sanity check on the parametrize matrix itself: somewhere in the test cases,
    at least one provider must produce shots from both teams. Otherwise the
    whole invariant test is vacuously skipped."""
    found_two_team_provider = False
    for _provider, loader, n_min in _ALL_CASES:
        try:
            actions, _ = loader()
        except Exception:  # noqa: S112 -- coverage check; loader failures are not the test's concern
            continue
        shots = actions[actions["type_id"] == spadlconfig.actiontype_id["shot"]]
        by_team = shots.groupby("team_id").size()
        if (by_team >= n_min).sum() >= 2:
            found_two_team_provider = True
            break
    assert found_two_team_provider, (
        "No provider in the parametrize matrix produced shots from >= 2 teams. "
        "The invariant test is vacuous; add a fixture that exercises both teams."
    )


# ---------------------------------------------------------------------------
# PR-S23 / silly-kicks 3.0.1: per-(team, period) orientation invariant.
# Closes the test-density gap that let the silly-kicks 3.0.0 native Sportec
# + Metrica per-period-absolute bug ship through PR-S22's invariant suite.
# Existing sample_match.parquet fixtures had only 2 shots / 1 period -- the
# per-period invariant could not be physically exercised until now.
# ---------------------------------------------------------------------------


_PER_PERIOD_CASES = [
    # (provider_label, loader, n_min_shots_per_team_period_group)
    ("sportec_native_per_period", _loaders.load_sportec_native_per_period, 3),
    ("metrica_native_per_period", _loaders.load_metrica_native_per_period, 3),
]


@pytest.mark.parametrize("provider,loader,n_min_shots_per_group", _PER_PERIOD_CASES)
def test_per_team_per_period_shots_attack_high_x(provider: str, loader, n_min_shots_per_group: int):
    """SPADL canonical LTR holds per-(period, team), not just in aggregate.

    A per-period-absolute bug (silly-kicks 3.0.0's Sportec + Metrica
    declaration error) is invisible to the aggregate test when one
    period's wrong-end signal cancels the other's. The per-(period,
    team) groupby exposes the bug.
    """
    actions, _home_team_id = loader()
    shots = actions[actions["type_id"].isin(_SHOT_TYPE_IDS)]
    assert len(shots) > 0, (
        f"{provider}: per-period fixture has no shot/shot_penalty/shot_freekick actions; "
        "regression in fixture or converter."
    )

    by_group = shots.groupby(["period_id", "team_id"]).agg(n=("start_x", "size"), mean_x=("start_x", "mean"))
    reliable = by_group[by_group["n"] >= n_min_shots_per_group]
    assert not reliable.empty, (
        f"{provider}: no (period, team) group has >= {n_min_shots_per_group} shots; "
        f"got {by_group.to_dict('index')}. Lower n_min or augment fixture."
    )

    failing = reliable[reliable["mean_x"] <= spadlconfig.field_length / 2]
    assert failing.empty, (
        f"{provider}: per-(period, team) orientation violated. Groups failing canonical LTR:\n"
        f"{failing.to_string()}\n"
        f"Expected: each (period, team) group's mean shot start_x > "
        f"{spadlconfig.field_length / 2}. Full breakdown:\n{by_group.to_string()}"
    )
