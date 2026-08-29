"""The C4 DSL asserts a count of action-coupled aggregators; nothing pinned it to the code.

There are TWO correct numbers and picking the wrong one is the likely failure:

    34  registered add_* in tracking.__all__   the ADR-051 mirror-registry surface
    33  action-coupled aggregators             what the C4 DSL sentence describes

They differ by `add_gradientsports_player_ids`, a jersey-number helper that enriches a roster and
is not coupled to an action. A maintainer who resolves the ambiguity by making the DSL quote the
LARGER number turns a true sentence false in a way no test would catch -- which is why this gate
names both.

The two figures above are ILLUSTRATIVE and were one behind the code when 4.79.0 landed (they read
33/32 while `__all__` already held 34). The gate itself derives both at runtime and was correct
throughout -- but a docstring whose whole job is stopping someone quoting the wrong number must not
itself quote a stale one. Prose beside a computed value goes stale; the computation does not.

Decision: Cycle B.
"""

from __future__ import annotations

import pathlib
import re

import silly_kicks.tracking as T

_DSL = pathlib.Path(__file__).resolve().parents[1] / "docs" / "c4" / "architecture.dsl"

#: Registered `add_*` helpers that are NOT action-coupled, each with a stated reason.
_NOT_ACTION_COUPLED: dict[str, str] = {
    "add_gradientsports_player_ids": (
        "jersey-number -> player_id helper. Enriches a ROSTER, takes no actions frame, and emits "
        "no per-action column, so it is not one of the aggregators the DSL sentence counts."
    ),
    "add_defending_gk_player_id": (
        "keeper-IDENTITY placement helper -- the apply-half of resolve_keeper_identities on the "
        "action grain. It stamps the resolved opponent-keeper id from a keeper_map; it takes NO "
        "frames, links no action to a frame, and computes no tracking-derived geometry, so it is "
        "not one of the spatial/GKDV tracking-feature aggregators the DSL sentence counts. Its "
        "per-action column is an identifier applied from a precomputed map, not a per-frame feature."
    ),
}


def _registered_add_star() -> set[str]:
    return {n for n in T.__all__ if n.startswith("add_")}


def test_the_dsl_aggregator_count_matches_the_code():
    registered = _registered_add_star()
    # Meta-assertion: a broken discovery would make this gate pass vacuously.
    assert len(registered) >= 25, f"discovery looks broken, found {sorted(registered)}"

    expected = len(registered) - len(_NOT_ACTION_COUPLED)
    text = _DSL.read_text(encoding="utf-8")
    found = re.search(r"(\d+) action-coupled aggregators", text)
    assert found is not None, "docs/c4/architecture.dsl no longer states an aggregator count"
    assert int(found.group(1)) == expected, (
        f"architecture.dsl says {found.group(1)} action-coupled aggregators; the code registers "
        f"{len(registered)} add_* of which {len(_NOT_ACTION_COUPLED)} are not action-coupled, so "
        f"the sentence should say {expected}. Do NOT resolve this by quoting {len(registered)} -- "
        f"that number is the ADR-051 mirror-registry surface, a different quantity."
    )


def test_not_action_coupled_entries_are_registered_helpers():
    """Self-burning-down: an exemption for a helper that no longer exists is stale scaffolding."""
    stale = sorted(set(_NOT_ACTION_COUPLED) - _registered_add_star())
    assert not stale, f"_NOT_ACTION_COUPLED names helpers that are not registered: {stale}"
