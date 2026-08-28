"""The FOV-observability completeness gate -- the anti-rot backstop for the ADR-077 cycle (Task 8).

Tasks 2--7 wired ``_observed_fraction`` / ``_observed_source`` companions across seven tracking
aggregators, one convex region per metric, through the ONE ``_fov_registry`` engine. THIS gate closes
the loop: every FOV-sensitive metric column must either carry a companion
(:func:`companioned_columns`) or be declared in :data:`_OBSERVABILITY_EXEMPT` with a stated reason.
A NEW region/aggregate metric therefore cannot ship un-annotated by accident -- it lands in
``required`` (derived structurally from the SB360 audit + the hand-curated
:data:`_AGGREGATE_FOV_SENSITIVE` bucket) and fails this gate until it is companioned or exempted.

**The two axes of ``required`` (R1).** The SB360 audit's ``region_support`` tag is a
SINGLE-PLAYER-PERTURBATION signal -- move one player, see if the metric moves. A mean-over-many
aggregate is robust to it (tagged ``no_support``) yet is genuinely biased by an FOV CROP (S1: a whole
cluster of players cut from view). The crop axis is a DIFFERENT axis the probe cannot reach, so
``required`` is the UNION of the two: the audit's ``region_support`` columns PLUS the hand-curated
``_AGGREGATE_FOV_SENSITIVE`` FOV-crop bucket.

**Scope -- the tracking ``add_*`` audit surface, boundary entries EXCLUDED (M2).** ``required`` is
derived ONLY from the tracking ``add_*`` audit surface (``public_add_star()`` == ``add_*`` in
``tracking.__all__``). The ADR-053/ADR-088 BOUNDARY entries (``gkdv.build_ghost_frames`` /
``gkdv.delta_das`` / ``gkdv.delta_threat_suppression`` / ``xtgk.compute_xt_gk_v2`` /
``spadl.add_restart_coordinates``) live OUTSIDE ``tracking.__all__`` -- they are enumerated in
``BOUNDARY_ENTRY_POINTS`` and carry a non-None ``verdict_provenance`` -- and are a COUNTERFACTUAL
boundary surface with their own observability story (``gkdv.build_ghost_frames`` even tags its
``ghost_x`` / ``ghost_y`` / ``displacement_m`` columns ``region_support``). Those columns are NOT part
of this cycle's region/area companion model and are excluded STRUCTURALLY (via the registry's own
``BOUNDARY_ENTRY_POINTS`` surface distinction), never by a name heuristic. See
:data:`_SCOPE_JUSTIFICATION`.

**Not asserted: that companions are a subset of ``required``.** ``companioned_columns()`` MAY EXCEED
``required`` -- unforced tight-ROI companions (``receiver_zone_density`` /
``defenders_in_triangle_to_goal``, both ``no_support`` in the audit) are allowed. The gate only
requires that ``required`` is fully covered, and that no EXEMPTION is stale (exempts must be a subset
of ``required``).
"""

from __future__ import annotations

import dataclasses

from silly_kicks.tracking._fov_registry import (
    _AGGREGATE_FOV_SENSITIVE,
    _OBSERVABILITY_EXEMPT,
    companioned_columns,
)
from tests.sb360._registry import (
    BOUNDARY_ENTRY_POINTS,
    SB360_ENTRIES,
    Sb360Entry,
    public_add_star,
)

#: The M2 scope justification the gate module documents -- read (not merely present) by
#: ``test_gate_scope_justification_present``, which also asserts each excluded category names a
#: REAL population in the registry, so the justification cannot be vacuously true.
#:
#: Two categories are excluded from ``required``:
#:
#: 1. ``support_data_defined`` columns (e.g. ``actor_arc_length_pre_window`` /
#:    ``actor_displacement_pre_window`` / ``elastic_confidence``) are TEMPORAL -- a pre-action window
#:    or a cross-frame confidence -- NOT an area integrated over a region, so a ``visible_area``
#:    observed-fraction is not the right observability model for them. They are neither
#:    ``region_support`` nor in ``_AGGREGATE_FOV_SENSITIVE``, so they never enter ``required``.
#: 2. The BOUNDARY-surface columns (``gkdv`` / ``xtgk`` v2 / ``spadl.add_restart_coordinates``) are a
#:    COUNTERFACTUAL boundary surface OUTSIDE ``tracking.__all__`` (``BOUNDARY_ENTRY_POINTS``, each
#:    carrying a non-None ``verdict_provenance``). ``gkdv.build_ghost_frames`` tags ``ghost_x`` /
#:    ``ghost_y`` / ``displacement_m`` ``region_support``, but they are NOT part of this cycle's
#:    region/area companion model -- they have their own observability story -- so they are excluded
#:    structurally by the registry's ``BOUNDARY_ENTRY_POINTS`` surface distinction.
_SCOPE_JUSTIFICATION: str = (
    "support_data_defined columns (actor_arc_length_pre_window / actor_displacement_pre_window / "
    "elastic_confidence) are temporal, not area, so a visible_area observed-fraction is not their "
    "observability model -- excluded from required. The gkdv / xtgk-v2 / spadl.add_restart_coordinates "
    "BOUNDARY entries (BOUNDARY_ENTRY_POINTS, outside tracking.__all__, verdict_provenance non-None) "
    "are a counterfactual boundary surface outside this cycle's region/area companion model -- their "
    "region_support columns (gkdv.build_ghost_frames ghost_x / ghost_y / displacement_m) are excluded "
    "structurally, never by a name heuristic."
)


# ---------------------------------------------------------------------------
# required = region_support (tracking add_* surface, boundary EXCLUDED) u _AGGREGATE_FOV_SENSITIVE
# ---------------------------------------------------------------------------
def _tracking_add_star_entries() -> dict[str, Sb360Entry]:
    """SB360 entries on the tracking ``add_*`` audit surface, BOUNDARY entries excluded.

    Structural, not a name heuristic: the surface is ``public_add_star()`` (== ``add_*`` in
    ``tracking.__all__``), and the boundary entries are exactly ``BOUNDARY_ENTRY_POINTS`` (which is
    DISJOINT from ``public_add_star()`` -- asserted in ``test_boundary_and_add_star_are_disjoint``).
    The ``not in BOUNDARY_ENTRY_POINTS`` clause is belt-and-suspenders on that disjointness.
    """
    adds = public_add_star()
    return {name: e for name, e in SB360_ENTRIES.items() if name in adds and name not in BOUNDARY_ENTRY_POINTS}


def _region_support_columns() -> set[str]:
    """Every ``region_support``-tagged column on the tracking ``add_*`` audit surface (boundary excluded)."""
    return {
        col
        for e in _tracking_add_star_entries().values()
        for col, tag in e.applicability.items()
        if tag == "region_support"
    }


def _required_columns() -> set[str]:
    """R1: BOTH axes -- the audit's perturbation-sensitive ``region_support`` UNION the hand-curated
    FOV-crop-sensitive ``_AGGREGATE_FOV_SENSITIVE`` bucket."""
    return _region_support_columns() | set(_AGGREGATE_FOV_SENSITIVE)


def _boundary_region_support_columns() -> set[str]:
    """``region_support`` columns living on the EXCLUDED boundary surface (kept OUT of ``required``)."""
    return {
        col
        for name, e in SB360_ENTRIES.items()
        if name in BOUNDARY_ENTRY_POINTS
        for col, tag in e.applicability.items()
        if tag == "region_support"
    }


def _support_data_defined_columns() -> set[str]:
    """Every ``support_data_defined``-tagged column on the tracking ``add_*`` surface (the M2 category)."""
    return {
        col
        for e in _tracking_add_star_entries().values()
        for col, tag in e.applicability.items()
        if tag == "support_data_defined"
    }


# ---------------------------------------------------------------------------
# The completeness gate
# ---------------------------------------------------------------------------
def test_boundary_and_add_star_are_disjoint():
    """The registry's OWN surface distinction: the boundary entry points are DISJOINT from the
    tracking ``add_*`` surface, so scoping ``required`` to ``public_add_star()`` excludes every
    boundary entry structurally (M2). This is what licenses the ``not in BOUNDARY_ENTRY_POINTS``
    exclusion being a structural fact rather than a name heuristic."""
    assert BOUNDARY_ENTRY_POINTS & public_add_star() == set()
    # Non-vacuity: there genuinely ARE boundary entries carrying region_support that the scoping
    # excludes -- otherwise the exclusion would be doing nothing.
    excluded = _boundary_region_support_columns()
    assert excluded, "no boundary region_support columns found -- the exclusion would be vacuous"
    assert excluded.isdisjoint(_region_support_columns())


def test_every_required_column_registered_or_exempt():
    """The gate: every FOV-sensitive column is companioned OR exempted, none left uncovered."""
    # Self-assert the population is non-empty: were both `_region_support_columns()` and
    # `_AGGREGATE_FOV_SENSITIVE` ever emptied, `missing` would be empty and the gate would pass
    # vacuously with nothing left to cover.
    assert _required_columns(), "required is empty -- the gate would pass vacuously"
    covered = companioned_columns() | set(_OBSERVABILITY_EXEMPT)  # covers maps RAW columns via `covers`
    missing = _required_columns() - covered
    assert not missing, f"FOV-sensitive columns with no companion or exemption: {sorted(missing)}"


def test_no_stale_exemption():
    """Every exemption is for a column that is actually REQUIRED -- no exemption for an unflagged
    column. (``companioned_columns()`` MAY exceed required -- that is NOT asserted here: extra
    tight-ROI companions are allowed unforced.)"""
    stale = set(_OBSERVABILITY_EXEMPT) - _required_columns()
    assert not stale, f"exemptions for non-required columns (stale): {sorted(stale)}"


def test_ghost_gk_columns_are_exempted_with_a_reason():
    """The two columns this task un-covers: ``add_ghost_gk``'s ``ghost_gk_x`` / ``ghost_gk_y`` are
    ``region_support`` (so REQUIRED) but carry no companion -- they must be EXEMPT with a non-empty
    reason. Landed RED (both absent from ``_OBSERVABILITY_EXEMPT``) before the exemptions existed."""
    rs = _region_support_columns()
    assert {"ghost_gk_x", "ghost_gk_y"} <= rs  # they ARE region_support, hence required
    for col in ("ghost_gk_x", "ghost_gk_y"):
        assert col in _OBSERVABILITY_EXEMPT, f"{col} is region_support but neither companioned nor exempt"
        reason = _OBSERVABILITY_EXEMPT[col]
        assert isinstance(reason, str) and reason.strip(), f"{col} exemption carries no reason"


def test_gate_scope_justification_present():
    """M2: the gate module documents WHY the excluded columns are excluded -- BOTH the
    ``support_data_defined`` (temporal, not area) exclusion AND the boundary-surface exclusion -- and
    each documented category names a REAL population in the registry, so the justification is not
    vacuously true."""
    j = _SCOPE_JUSTIFICATION.lower()

    # (1) The justification documents the support_data_defined (temporal) exclusion, naming examples.
    assert "support_data_defined" in j
    assert "actor_arc_length_pre_window" in j or "actor_displacement_pre_window" in j
    assert "elastic_confidence" in j
    assert "temporal" in j and "not area" in j

    # (2) The justification documents the boundary-surface exclusion, naming the mechanism.
    assert "boundary" in j
    assert "gkdv" in j
    assert "structural" in j  # excluded structurally, never by a name heuristic

    # (3) Non-vacuity: each documented category actually EXISTS in the registry.
    #     (3a) the named support_data_defined examples really carry that tag.
    sdd = _support_data_defined_columns()
    assert {"actor_arc_length_pre_window", "actor_displacement_pre_window", "elastic_confidence"} <= sdd
    #     None of them is region_support / aggregate -- so they legitimately never enter `required`.
    assert sdd.isdisjoint(_required_columns())
    #     (3b) the named boundary region_support columns really exist and are excluded from `required`.
    boundary_rs = _boundary_region_support_columns()
    assert {"ghost_x", "ghost_y", "displacement_m"} <= boundary_rs
    assert boundary_rs.isdisjoint(_required_columns())


def test_non_vacuity_plant():
    """M3: the detector fires for a NEW required column -- so the gate cannot pass merely by knowing
    today's population. A synthetic column spelled to appear in NO committed entry is (a) NOT covered,
    and (b) WOULD be flagged missing if it were required."""
    synthetic = "zzz_synthetic_fov_probe_col"
    covered = companioned_columns() | set(_OBSERVABILITY_EXEMPT)

    # It is not silently already-covered.
    assert synthetic not in covered
    assert synthetic not in _required_columns()  # and not silently already-required

    # If it WERE required, the gate's own `missing` computation would flag it -- i.e. the detector
    # fires. This is the same expression `test_every_required_column_registered_or_exempt` evaluates,
    # with the synthetic column injected into the population.
    missing_if_required = ({synthetic} | _required_columns()) - covered
    assert synthetic in missing_if_required


def test_non_vacuity_plant_exercises_the_derivation(monkeypatch):
    """M3 (stronger, ADR-056): plant a synthetic ``region_support`` entry INTO the registry the
    DERIVATION reads (``SB360_ENTRIES``) and assert the structural ``applicability``-read derivation
    picks it up AND the gate's real ``missing`` computation flags it -- locking the
    derivation -> required -> missing path end-to-end.

    ``test_non_vacuity_plant`` above injects into the ``required`` SET, so it only tests set
    arithmetic and is BLIND to the exact ADR-056 failure mode: a refactor replacing the structural
    ``applicability``-read ``_region_support_columns()`` with a hardcoded column list. This plant
    would go RED under that refactor, because assertion (a) below fails the moment the derivation
    stops reading ``applicability``.
    """
    synthetic = "zzz_synthetic_region_support_col"
    # A real tracking add_* entry the derivation actually iterates. Preserve its existing
    # applicability (dataclasses.replace) and ADD the synthetic region_support tag, so the plant is
    # purely ADDITIVE -- every real region_support column still resolves alongside the synthetic one.
    name = next(iter(sorted(public_add_star() & set(SB360_ENTRIES))))
    original = SB360_ENTRIES[name]
    planted = dataclasses.replace(original, applicability={**original.applicability, synthetic: "region_support"})
    monkeypatch.setitem(SB360_ENTRIES, name, planted)

    # (a) the DERIVATION (structural applicability-read) picks the synthetic column up -- this is the
    #     assertion a hardcoded-list refactor would break.
    assert synthetic in _region_support_columns()
    assert synthetic in _required_columns()

    # (b) the gate's real `missing` computation (the identical expression the live gate evaluates)
    #     flags it: it is region_support (required) yet neither companioned nor exempt.
    covered = companioned_columns() | set(_OBSERVABILITY_EXEMPT)
    missing = _required_columns() - covered
    assert synthetic in missing
