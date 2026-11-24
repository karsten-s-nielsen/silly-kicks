"""The W9.6 game-split property: two games in one call equal the two games called separately.

Spec 2026-11-08 tracking-batch §4.9 W9.6 (R5): every public, non-exempt population member, on ``links=None`` and on
caller-supplied ``links=``. The two games share ``period_id``, ``frame_id`` and ``action_id`` and their clocks are
offset, so a game-blind join links one game's actions to the other game's frames. Every xfn factory and default xfn
list runs through the public VAEP feature path, whose game-state slots carry duplicate ``action_id``s.

A "member raised" red counts only when the same call succeeded on both single-game parts and the raise came from the
two-game call itself, with its stack inside ``silly_kicks/`` (A2-PLAN-50, A3-PLAN-67). Everything else -- the
resolver, the single-game parts, the linker, the comparators -- runs outside the ``try``, so a scaffolding defect is a
hard failure, never an expected red.
"""

from __future__ import annotations

import traceback

import pytest

from silly_kicks.tracking.utils import link_actions_to_frames
from tests._prod_fixtures import make_two_game_inputs
from tests.invariants._link_population import (
    COMPARATORS,
    accepts_links,
    classify_member_raise,
    comparator_for,
    member_actions,
    public_non_exempt,
    resolve_call,
    split_by_game,
)
from tests.invariants.link_population_registry import (
    ADAPTERS,
    LINKS_REQUIRED,
)

PUBLIC_NON_EXEMPT = public_non_exempt()


def _takes_links(member: str) -> bool:
    return member in ADAPTERS or accepts_links(member)


def _whole_and_parts(member, actions, frames, links_mode):
    (a, fa), (b, fb) = split_by_game(actions, frames)
    call = resolve_call(member)  # scaffolding: outside any try, so its errors are hard failures

    def links_for(acts, frs):
        if links_mode == "none":
            return None
        return link_actions_to_frames(member_actions(member, acts), frs, on_low_coverage="ignore")[0]

    parts = [call(a, fa, links_for(a, fa)), call(b, fb, links_for(b, fb))]  # well-formed-call proof (single game)
    links = links_for(actions, frames)
    try:
        whole = call(actions, frames, links)  # only the two-game member call is inside the try
    except Exception as exc:
        origin = classify_member_raise(traceback.extract_tb(exc.__traceback__))  # pytest.fail if no package frame
        raise AssertionError(f"{member} raised {type(exc).__name__} via {origin.filename}:{origin.lineno}") from exc
    return whole, parts


@pytest.mark.parametrize("member", [m for m in PUBLIC_NON_EXEMPT if m not in LINKS_REQUIRED])
def test_game_split_links_none(member):
    actions, frames = make_two_game_inputs()
    whole, parts = _whole_and_parts(member, actions, frames, "none")
    COMPARATORS[comparator_for(member)].assert_split_equal(whole, parts, actions)


@pytest.mark.parametrize("member", [m for m in PUBLIC_NON_EXEMPT if _takes_links(m)])
def test_game_split_caller_links(member):  # spec W6 and W9 acceptance: caller-supplied links= on two games
    actions, frames = make_two_game_inputs()
    whole, parts = _whole_and_parts(member, actions, frames, "caller")
    COMPARATORS[comparator_for(member)].assert_split_equal(whole, parts, actions)
