"""The single-sourced SB360 call convention (``scripts/_sb_battery.py``).

Proves the round-4 de-fork: the audit (``tests/sb360``) and the licensed-corpus driver resolve ONE
``ADAPTER_MAP`` / ``call_aggregator``, the population matches the audit's ``public_add_star()``, and
the module is a clean leaf (imports no ``tests``).
"""

from __future__ import annotations

import ast
import inspect

import scripts._sb_battery as B
from tests.sb360 import _fixture as F
from tests.sb360 import _registry as R


def test_population_matches_registry_public_add_star():
    # ONE enumeration: the driver and the audit see the same aggregator set.
    assert set(B.registered_add_star_aggregators()) == R.public_add_star()


def test_adapter_map_is_single_sourced_in_sb_battery():
    R._init_adapters()
    # The audit's map IS the shared map (content-equal; _adapters() returns a defensive copy).
    assert R._adapters() == B.ADAPTER_MAP
    # ...and every adapter is defined HERE, not a stale local copy left behind by the move.
    for name, fn in B.ADAPTER_MAP.items():
        assert fn.__module__ == "scripts._sb_battery", f"{name} resolves to {fn.__module__}"


def test_call_aggregator_action_context_emits_columns():
    actions, frames, links = F.build_leg_a()
    result = B.call_aggregator("add_action_context", actions, frames, links, F.HOME_TEAM_ID)
    added = [c for c in result.columns if c not in actions.columns]
    assert added, "add_action_context emitted no columns through the shared call convention"


def test_every_aggregator_resolves_without_a_swallowed_adapter_typeerror():
    """Driver-side mirror of tests/sb360/test_registry_surface.py:174-192.

    Every registered aggregator is attempted, and no aggregator falls off the generic path with a
    call-shape ``TypeError`` (the silent-``cols=()`` defect). A genuine library REFUSAL on
    freeze-frames (e.g. a KeyError) is a legitimate recorded result; an adapter mis-call is not.
    """
    actions, frames, links = F.build_leg_a()
    battery = B.run_add_star_battery(actions, frames, links=links, home_team_id=F.HOME_TEAM_ID)
    assert set(battery) == set(B.registered_add_star_aggregators())
    for name, res in battery.items():
        if isinstance(res, str):
            assert res.startswith("raises:"), f"{name}: unexpected marker {res!r}"
            assert "unexpected keyword argument" not in res, f"{name}: adapter mis-call, not a real refusal: {res}"
            assert "positional argument" not in res, f"{name}: adapter mis-call, not a real refusal: {res}"


def test_sb_battery_is_a_leaf_imports_no_tests_module():
    """The invariant that keeps layering one-directional: scripts/_sb_battery imports no tests."""
    tree = ast.parse(inspect.getsource(B))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    leaked = sorted(m for m in imported if m == "tests" or m.startswith("tests."))
    assert not leaked, f"scripts/_sb_battery must not import tests modules: {leaked}"
