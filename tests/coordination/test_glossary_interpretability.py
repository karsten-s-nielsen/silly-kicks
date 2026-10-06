"""Spec §5 goal 4 (amended 2026-10-02) + §2: every coordination glossary entry is readable by a first-time user.

Each metric entry states its scale and direction (a published reference value where the cited paper reports one, spec
§8.5), and each is attributed to the source that supplies its method (spec §2) -- the M-7 misattributions (Pfister on
vector coding, Folgado on the relative stretch index, Welch on the median frequency) cannot come back.
"""

from __future__ import annotations

import pytest

from silly_kicks import feature_glossary as fg

_ENTRIES = {name: fc for name, fc in fg.FEATURE_GLOSSARY.items() if "coordination" in (fc.emitting_module or "")}
_METRICS = {name: fc for name, fc in _ENTRIES.items() if not fc.definition.startswith("Coverage:")}

#: Phrases that state a value's scale: a bounded interval, a sign convention, or a physical unit.
_SCALE_MARKERS = (
    "[0, 1]",
    "[-1, 1]",
    "[0, 0.5]",
    "[0, 360)",
    "(-180 to 180)",
    ">= 0",
    "in metres",
    "in seconds",
    "cycles per minute",
    "in the metric's own units",
    "(dimensionless)",
)


def test_the_population_is_non_empty():
    assert len(_ENTRIES) >= 70 and len(_METRICS) >= 55  # non-vacuity: the coordination glossary is all here


@pytest.mark.parametrize("name", sorted(_METRICS))
def test_every_metric_entry_states_its_scale(name):
    definition = _METRICS[name].definition
    assert any(marker in definition for marker in _SCALE_MARKERS), definition


@pytest.mark.parametrize("name", sorted(n for n in _METRICS if n.endswith("_percentile")))
def test_percentile_entries_state_scale_and_direction(name):
    definition = _METRICS[name].definition
    assert "[0, 1]" in definition and "higher =" in definition and "0.5" in definition


@pytest.mark.parametrize("name", sorted(n for n in _METRICS if n.endswith("_excess")))
def test_excess_entries_state_direction(name):
    assert "> 0 = above chance" in _METRICS[name].definition


def _expected_attribution(name: str) -> str | None:
    if name == "coord_rsi_bimodality_coefficient":
        return fg._A_TF58_PFISTER
    if name.startswith("coord_rsi_"):
        return fg._A_TF58_BOURBOUSSON
    if name.startswith("coord_vc_"):
        return fg._A_TF58_VECTOR_CODING
    if name.startswith("coord_rp_pct_near_in_phase"):
        return fg._A_TF58_FOLGADO
    if name in ("coord_rp_mean_deg", "coord_rp_resultant_length", "coord_rp_circ_sd_deg"):
        return fg._A_TF58_RP_CIRCULAR  # B m9: the circular statistics also credit Mardia & Jupp
    if name.startswith("coord_rp_"):
        return fg._A_TF58_BOURBOUSSON
    if name == "coord_median_freq_cpm":
        return fg._A_TF58_MOURA_2013
    if name.startswith("coord_coh_"):
        return fg._A_TF58_COHERENCE
    if name.startswith("coord_xc_"):
        return fg._A_TF58_MOURA_2016
    # A-38 / review P-7: the cluster-phase, SampEn and team-sync families must be attributed too, not left to the
    # None fall-through (the gate skipped them). SampEn/Cross-SampEn -> Richman & Moorman; the Kuramoto cluster-phase
    # statistics -> Richardson & Frank, except rho SD and the team-team Pearson r -> Duarte (who applied them to
    # football); checked in that order so the sampen and rho-SD special cases win over the family prefix.
    if name.endswith("_sampen"):
        return fg._A_TF58_RICHMAN
    if name == "coord_rho_group_sd":
        return fg._A_TF58_DUARTE
    if name.startswith("coord_rho_") or name.startswith("coord_phi_"):
        return fg._A_TF58_RICHARDSON
    if name.startswith("coord_team_sync_"):
        return fg._A_TF58_DUARTE
    return None


@pytest.mark.parametrize("name", sorted(n for n in _METRICS if _expected_attribution(n) is not None))
def test_attribution_is_the_source_that_supplies_the_method(name):
    assert _METRICS[name].attribution == _expected_attribution(name)


def test_the_vector_coding_token_names_the_coupling_angle_sources():
    assert "Sparrow et al. (1987)" in fg._A_TF58_VECTOR_CODING
    assert "Chang, Van Emmerik & Hamill (2008)" in fg._A_TF58_VECTOR_CODING


def test_pfister_is_cited_for_the_bimodality_coefficient_only():
    cited = sorted(n for n, fc in _ENTRIES.items() if fc.attribution == fg._A_TF58_PFISTER)
    assert cited == ["coord_rsi_bimodality_coefficient"]
