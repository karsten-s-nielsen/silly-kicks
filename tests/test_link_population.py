"""The W9 link population registry is derived and asserted exactly (spec 2026-11-08 tracking-batch W9.6; ADR-056)."""

from __future__ import annotations

import pathlib
import traceback

import pytest

from tests.invariants import _link_population as L
from tests.invariants.link_population_registry import (
    ADAPTERS,
    BASE_MEMBERS,
    EXEMPT_POOLED,
    LINK_POPULATION,
    PER_KEY,
    PREREQUISITE_ADAPTERS,
    REFERENCE_ONLY,
)

#: Owner ruling 2026-11-09: the derived population (284) is accepted over the spec's 150 stop line. Growth beyond it
#: goes back to the owner (the spec stop rule), so the bound is the accepted size. Owner ruling 2026-11-09 (later):
#: raised to 287 for three private stage functions of the W0 driver itself (spec R21-R23 rework); no package member.
OWNER_ACCEPTED_POPULATION = 287

#: Reviewer A's scan (spec W9.6): public functions reachable only through _resolve_action_frame_context.
_REVIEWER_A_TWENTY = tuple(
    f"silly_kicks.{pkg}tracking.features:{name}"
    for pkg in ("", "atomic.")
    for name in (
        "nearest_defender_distance",
        "actor_speed",
        "receiver_zone_density",
        "defenders_in_triangle_to_goal",
        "pre_shot_gk_x",
        "pre_shot_gk_y",
        "pre_shot_gk_distance_to_goal",
        "pre_shot_gk_distance_to_shot",
        "pre_shot_gk_angle_to_shot_trajectory",
        "pre_shot_gk_angle_off_goal_line",
    )
)


def test_population_matches_registry():
    assert L.derive_population() == LINK_POPULATION


def test_underivable_empty():
    assert set(LINK_POPULATION) - L.derive_population() == set()


def test_base_members_frozen():
    # the population as derived at BASE_SHA; the W0 driver's --members runs exactly these on both legs (TB-PLAN-27)
    assert BASE_MEMBERS == LINK_POPULATION


def test_every_private_member_covered():
    coverage = L.derive_coverage()
    private = {m for m in LINK_POPULATION if m.startswith("silly_kicks") and not L.is_public(m)}
    assert private - set(coverage) == set(REFERENCE_ONLY)
    g = L.graph()
    for member in REFERENCE_ONLY:
        assert not {c for c in g.callers(member) if c.startswith(("silly_kicks", "scripts"))}, member


def test_adapters_exactly_nonconforming():
    # plus the ADR-078-shaped members whose documented prerequisite runs first (each with its reason)
    assert set(ADAPTERS) == L.nonconforming_members() | set(PREREQUISITE_ADAPTERS)


def test_adapter_split():
    public = [m for m in LINK_POPULATION if L.is_public(m) and m not in EXEMPT_POOLED]
    routes = {m: L.route(m) for m in public}
    assert set(routes.values()) <= {"battery", "generic", "adapter"}
    assert {m for m, r in routes.items() if r == "adapter"} == set(ADAPTERS)
    # non-vacuity: every route is used
    assert {"battery", "generic", "adapter"} == set(routes.values())


def test_cannot_fail_entries_name_public_members():
    from tests.invariants.link_population_registry import CANNOT_FAIL_AT_BASE

    for variant, member in CANNOT_FAIL_AT_BASE:
        assert variant in ("links_none", "caller_links") and member in LINK_POPULATION and L.is_public(member)


def test_green_after_linker_entries_name_public_members():
    from tests.invariants.link_population_registry import CANNOT_FAIL_AT_BASE, GREEN_AFTER_LINKER

    for variant, member in GREEN_AFTER_LINKER:
        assert variant in ("links_none", "caller_links") and member in LINK_POPULATION and L.is_public(member)
    assert not set(GREEN_AFTER_LINKER) & set(CANNOT_FAIL_AT_BASE)  # a pair is red at the base or it is not


def test_registries_name_members():
    assert set(EXEMPT_POOLED) <= LINK_POPULATION
    assert set(PER_KEY) <= LINK_POPULATION
    assert set(REFERENCE_ONLY) <= LINK_POPULATION


def test_population_bound():
    assert len(LINK_POPULATION) <= OWNER_ACCEPTED_POPULATION


#: Owner ruling 2026-11-11: the link-identity functions are link machinery, the same role as ``LINKERS[:2]``. The
#: seed subtracts exactly these names. Owner ruling 2026-11-11 (later): ``require_links_bound`` is the sixth, the
#: entry check every public ``links=`` acceptor calls. A further name goes back to the owner (the spec stop rule).
_OWNER_ACCEPTED_MACHINERY = frozenset(
    f"silly_kicks.tracking._link_identity:{name}"
    for name in (
        "resolve_link_positions",
        "remap_links",
        "_remap",
        "_verify_positions",
        "_legacy_positions",
        "require_links_bound",
    )
)


def test_link_machinery_excluded_by_name():
    assert len(L.LINK_MACHINERY) == len(set(L.LINK_MACHINERY))
    assert set(L.LINK_MACHINERY) == _OWNER_ACCEPTED_MACHINERY
    fns = L.graph().functions()
    population = L.derive_population()
    for member in L.LINK_MACHINERY:
        # non-vacuity: the seed rule ("a ``links`` parameter") selects the member, so the subtraction removes it
        assert member in fns and "links" in L._params(fns[member]), member
        assert member not in population, member  # the fixpoint does not bring it back


def test_reviewer_a_twenty_present():
    assert len(_REVIEWER_A_TWENTY) == 20
    assert set(_REVIEWER_A_TWENTY) <= LINK_POPULATION


def test_classify_member_raise():
    pkg = L.PACKAGE_ROOT / "x.py"
    test_frame = traceback.FrameSummary("tests/test_x.py", 1, "t")
    pandas_frame = traceback.FrameSummary("pandas/core/indexes/base.py", 4436, "reindex")
    hit = traceback.FrameSummary(str(pkg), 7, "f")
    assert L.classify_member_raise([test_frame, hit, pandas_frame]) is hit
    # the other separator form: forward slashes on Windows, where str(pkg) uses backslashes (same path on POSIX)
    native = traceback.FrameSummary(pkg.as_posix(), 7, "f")
    assert L.classify_member_raise([test_frame, native, pandas_frame]) is native
    with pytest.raises(pytest.fail.Exception):
        L.classify_member_raise([test_frame, traceback.FrameSummary("scripts/_sb_battery.py", 3, "c"), pandas_frame])
    assert not pathlib.Path("scripts/_sb_battery.py").resolve().is_relative_to(L.PACKAGE_ROOT)


def test_split_comparator_compares_slices_not_a_concat():
    """pandas ``concat`` turns an object column of ``pd.NA``/``None`` into ``nan`` (all-NA entries), so a
    comparator that concatenates the per-game parts reports a difference the member never made. Each part is
    compared with its own slice of the whole instead."""
    import pandas as pd

    part = pd.DataFrame({"reason": pd.array([pd.NA, None, pd.NA], dtype=object), "x": [1.0, 2.0, 3.0]})
    whole = pd.concat([part, part], ignore_index=True)
    whole["reason"] = pd.array([pd.NA, None, pd.NA] * 2, dtype=object)  # the values the member really returns
    L.COMPARATORS["per_action"].assert_split_equal(whole, [part, part], whole)
    changed = whole.copy()
    changed.loc[4, "x"] = 9.0
    with pytest.raises(AssertionError):  # non-vacuity: a real difference in the second part still fails
        L.COMPARATORS["per_action"].assert_split_equal(changed, [part, part], changed)
