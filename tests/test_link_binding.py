"""W9.5 binding fixtures: caller-supplied ``links=`` bind to exactly the ``actions`` frame they were built on.

Spec 2026-11-08 tracking-batch §4.9 W9.5 and W9.6, plan Task 4 step 2.

- **Caller mismatch** (two games): the caller passes a frame that is not the one its links were built on -- a mart
  order, the games reversed, or a subset. Every ``links=`` acceptor must raise the W9.5 ``ValueError``.
- **Internal re-sort** (one game, shuffled rows): a function that re-sorts ``actions`` and passes ``links`` on must
  remap them. Its values with matching ``links`` equal the clean call (``links=None``) bit for bit.
- **Planted mutations** prove each case kind is red without its own protection (TB-PLAN-15): ``no_verify`` removes
  the position check, ``forward_unchanged`` makes ``remap_links`` forward the caller's pointers unchanged.

Task 5 migrated the consumers onto ``resolve_link_positions`` and ``remap_links`` (and the entry check
``require_links_bound``), so every case is green; the Task 3/4 ``xfail`` marks are gone.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from scripts._bitwise import assert_bits_equal
from silly_kicks.tracking import _link_identity
from silly_kicks.tracking.utils import link_actions_to_frames
from tests._prod_fixtures import GAME_A, GAME_B, make_single_game_inputs, make_two_game_inputs
from tests.invariants._link_population import accepts_links, member_actions, public_non_exempt, resolve_call
from tests.invariants.link_population_registry import ADAPTERS

_BINDING = "links must be built from exactly this actions frame"
CALLER_CASES = ("mart", "reversed_games", "subset")


def _binds(member: str) -> bool:
    """True when the member's OWN signature takes caller ``links`` AND an ``actions`` frame: only then can it
    check that the links were built on that frame (spec W9.5). A member reached through an adapter is judged by
    its own signature too: an xfn factory, a per-series function, or a ``links``-only scorer (``sync_score``)
    has no frame to bind the links to."""
    import inspect

    from tests.invariants._link_population import resolve_object

    obj = resolve_object(member)
    if not callable(obj):  # a default xfn list (a module-level name): no signature, no frame to bind
        return False
    params = inspect.signature(obj.__init__ if isinstance(obj, type) else obj).parameters
    return "links" in params and "actions" in params


#: Every public, non-exempt population member whose own signature takes caller links and an actions frame.
ACCEPTORS = tuple(m for m in public_non_exempt() if _binds(m))

#: The re-sort sites (spec W9.5): each re-sorts, filters, or subsets ``actions`` and then passes ``links`` on, through
#: ``remap_links``. From the W9.6 census (spec section 4.9 consumer table, Task 5). ``resolve_restart_geometry`` is
#: the CALLEE of the ``add_restart_coordinates`` re-sort (spadl/utils), not a re-sort site itself, so the site is
#: listed; the callee stays covered as an acceptor.
RESORT_SITES = (
    "silly_kicks.atomic.spadl.utils:add_pre_shot_gk_context",
    "silly_kicks.spadl.utils:add_pre_shot_gk_context",
    "silly_kicks.spadl.utils:add_restart_coordinates",
    "silly_kicks.tracking.defensive_credit._orchestration:compute_defensive_credits",
    "silly_kicks.restdefense._windows:select_rest_defense_samples",
    "silly_kicks.coordination._windows:possession_windows_from_actions",
)

#: The member whose caller-mismatch cases the mutation pairing runs on (any acceptor exercises the shared check).
REPRESENTATIVE_ACCEPTOR = "silly_kicks.tracking.features:add_action_context"


def _mismatched(case: str, actions: pd.DataFrame) -> pd.DataFrame:
    """The frame a caller passes with links that were built on the two-game ``actions``."""
    is_a = (actions["game_id"] == GAME_A).to_numpy()
    if case == "mart":  # a data mart's order: by action_id, the later game first on each id
        out = actions.sort_values(["action_id", "game_id"], ascending=[True, False], kind="stable")
    elif case == "reversed_games":
        out = pd.concat([actions[~is_a], actions[is_a]])
    elif case == "subset":
        out = actions[is_a]
    else:
        raise ValueError(case)
    return out.reset_index(drop=True)


def _expect_binding_error(call: Callable[[], object]) -> None:
    """Pass only on the W9.5 ``ValueError``; every other outcome is an ``AssertionError`` (the red the marks expect)."""
    try:
        call()
    except ValueError as exc:
        if _BINDING not in str(exc):
            raise AssertionError(f"ValueError without the W9.5 binding message: {exc}") from exc
        return
    except Exception as exc:  # a wrong join that crashes somewhere else is red, never green
        raise AssertionError(f"{type(exc).__name__} instead of the W9.5 ValueError: {exc}") from exc
    raise AssertionError("no ValueError: the mismatched links were accepted")


def _caller_case(member: str, case: str) -> None:
    actions, frames = make_two_game_inputs()
    links = link_actions_to_frames(member_actions(member, actions), frames, on_low_coverage="ignore")[0]
    call = resolve_call(member)
    _expect_binding_error(lambda: call(_mismatched(case, actions), frames, links))


def _helper_case(case: str) -> None:
    """The identity helper's own caller-mismatch case: ``resolve_link_positions`` on the mismatched frame."""
    actions, frames = make_two_game_inputs()
    links = link_actions_to_frames(actions, frames, on_low_coverage="ignore")[0]
    _expect_binding_error(lambda: _link_identity.resolve_link_positions(_mismatched(case, actions), links))


def _shuffled_single_game(member: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One game with the member's own rows out of order, so a re-sort inside the member is a real permutation.

    The shuffle comes after ``member_actions``: the atomic converter reads chronological SPADL rows only (ADR-065),
    so an atomic member gets its atomic frame shuffled, not a conversion of shuffled SPADL rows.
    """
    actions, frames = make_single_game_inputs()
    own = member_actions(member, actions)
    perm = np.random.default_rng(20261110).permutation(len(own))
    assert not np.array_equal(perm, np.arange(len(own)))
    return own.iloc[perm].reset_index(drop=True), frames


def _resort_case(member: str) -> None:
    """The member's result on shuffled rows with matching links equals its result on the rows in order with
    THEIR matching links: the re-sort inside the member remaps the pointers (W9.5). The reference is a call
    with links, not the clean ``links=None`` call, because caller links change the values of some members by
    design (``possession_windows_from_actions`` takes the frame time of a linked action)."""
    actions, frames = _shuffled_single_game(member)
    in_order, _ = _ordered_single_game(member)
    call = resolve_call(member, member_frame=True)
    reference = call(in_order, frames, link_actions_to_frames(in_order, frames, on_low_coverage="ignore")[0])
    links = link_actions_to_frames(actions, frames, on_low_coverage="ignore")[0]
    try:
        with_links = call(actions, frames, links)
    except Exception as exc:  # a raise on matching links is a failure (plan Task 4 step 2)
        raise AssertionError(f"{member} raised {type(exc).__name__} on matching links: {exc}") from exc
    assert_bits_equal(with_links, reference)


def _ordered_single_game(member: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The same game as :func:`_shuffled_single_game`, with the member's own rows in their original order."""
    actions, frames = make_single_game_inputs()
    return member_actions(member, actions).reset_index(drop=True), frames


@pytest.fixture
def planted(monkeypatch) -> Callable[[str], None]:
    """Install one planted mutation in ``_link_identity`` for the rest of the test."""

    def install(mode: str) -> None:
        if mode == "no_verify":
            monkeypatch.setattr(_link_identity, "_verify_positions", lambda actions, links, rows: None)
        elif mode == "forward_unchanged":
            monkeypatch.setattr(_link_identity, "_remap", lambda links, permutation, new_actions: links)
        else:
            raise ValueError(mode)

    return install


@pytest.mark.parametrize("case", CALLER_CASES)
def test_helper_rejects_caller_mismatch(case):
    _helper_case(case)


@pytest.mark.parametrize("case", CALLER_CASES)
@pytest.mark.parametrize("member", ACCEPTORS)
def test_caller_mismatch_raises(member, case):
    _caller_case(member, case)


@pytest.mark.parametrize("member", RESORT_SITES)
def test_internal_resort_equals_clean_call(member):
    _resort_case(member)


@pytest.mark.parametrize("member", RESORT_SITES)
def test_resort_fixture_is_a_real_permutation(member):
    actions, _ = _shuffled_single_game(member)
    assert not actions["time_seconds"].is_monotonic_increasing


#: (case id, mutation that must turn it red, the case's check). See the module docstring for the pairing.
_PAIRINGS: list = [
    *[(f"helper-{c}", "no_verify", lambda c=c: _helper_case(c)) for c in CALLER_CASES],
    *[(f"member-{c}", "no_verify", lambda c=c: _caller_case(REPRESENTATIVE_ACCEPTOR, c)) for c in CALLER_CASES],
    *[(f"resort-{m}", "forward_unchanged", lambda m=m: _resort_case(m)) for m in RESORT_SITES],
]


@pytest.mark.parametrize(("case", "mutation", "check"), _PAIRINGS, ids=lambda v: v if isinstance(v, str) else "")
def test_mutation_turns_fixture_red(case, mutation, check, planted):
    check()  # green without the mutation: the pairing proves a protection, not a broken fixture
    planted(mutation)
    try:
        check()
    except AssertionError:
        return
    raise AssertionError(f"{case} stayed green under the {mutation} mutation")


def test_representative_acceptor_is_an_acceptor():
    assert REPRESENTATIVE_ACCEPTOR in ACCEPTORS
    assert set(RESORT_SITES) <= set(ACCEPTORS)


def test_acceptors_are_every_links_taker_with_an_actions_frame():
    """Anti-rot: the derivation keeps every public member that takes ``links=`` (directly or through an adapter
    that forwards them) unless its own signature has no actions frame to bind them to."""
    takers = {m for m in public_non_exempt() if m in ADAPTERS or accepts_links(m)}
    assert set(ACCEPTORS) <= takers
    dropped = takers - set(ACCEPTORS)
    assert all(not _binds(m) for m in dropped)
    assert len(ACCEPTORS) >= 60, len(ACCEPTORS)  # non-vacuity: the population's acceptors, not a handful


def test_mismatch_cases_change_the_frame():  # non-vacuity: each case is a different frame from the links' frame
    actions, _ = make_two_game_inputs()
    for case in CALLER_CASES:
        other = _mismatched(case, actions)
        assert not (len(other) == len(actions) and other.equals(actions)), case
    assert set(_mismatched("subset", actions)["game_id"]) == {GAME_A}
    assert _mismatched("reversed_games", actions)["game_id"].iloc[0] == GAME_B
