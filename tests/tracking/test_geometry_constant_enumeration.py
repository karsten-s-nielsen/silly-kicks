"""Completeness by ENUMERATION, not by remembering to declare (ADR-050).

Every geometry-named module constant in a model's extractor must either be DECLARED in that
model's feature contract -- so a change to it makes ``load()`` raise -- or be explicitly EXEMPT
with a reason. Adding a name to the exemption list is a visible code-review decision; that is the
forcing function. A gate that relies on the next contributor remembering is not a gate.
"""

from __future__ import annotations

import ast
from pathlib import Path

SK = Path(__file__).resolve().parents[2] / "silly_kicks"

_MODULES = [
    SK / "tracking" / "_ghost_gk.py",
    SK / "tracking" / "_xshot_occurrence.py",
    SK / "tracking" / "_xcross_attempt.py",
    SK / "tracking" / "defensive_credit" / "_params.py",
]

_GEOMETRY_NAME = ("PENALTY", "BOX", "GOAL", "PITCH", "FIELD", "AREA")

#: Module-level geometry constants deliberately NOT in any contract, each with a reason.
#: Adding a name here is a visible code-review decision -- the forcing function.
#:
#: THE RULE for a DERIVED constant, because the two cases look identical otherwise:
#:   * derived from a DECLARED constant  -> map it to that constant's key in
#:     DECLARED_CONSTANT_SOURCES. (GOAL_Y_MIN/GOAL_Y_MAX = GOAL_Y_CENTRE -/+ GOAL_WIDTH/2, so they
#:     move iff goal_width moves -- declared.)
#:   * derived from PITCH DIMENSIONS    -> exempt here. (GOAL_Y_CENTRE/_GOAL_Y/_GOAL_Y_C are all
#:     just PITCH_WIDTH/2; they are already covered by the pitch_length/pitch_width fail-closed
#:     guard, so declaring them would double-count one quantity under two names.)
#: Without this written down, the next person extending either list follows whichever precedent
#: they happen to read first -- and both precedents are present below.
_EXEMPT = {
    "_FIELD_LENGTH": "pitch dimension, covered by the pitch_length/pitch_width fail-closed guard",
    "_FIELD_WIDTH": "pitch dimension, covered by the pitch_length/pitch_width fail-closed guard",
    "_GOAL_Y_C": "goal centre y, derived as _FIELD_WIDTH/2; no independent value",
    "_GOAL_Y": "goal centre y, derived as _FIELD_WIDTH/2; no independent value",
    "GOAL_Y_CENTRE": "goal centre y, derived as PITCH_WIDTH/2; no independent value",
    "GHOST_GK_GOAL_END_UNRESOLVED": (
        "NOT a geometry constant: a provenance STRING in the closed ghost_gk_source vocabulary "
        "(ADR-055). It matches only because the name-based enumerator keys on the substring "
        "'GOAL', and it carries no numeric value a feature contract could declare or a probe "
        "could make load-bearing. Its sibling tokens (GHOST_GK_NO_KEEPER, GHOST_GK_UNLINKED, ...) "
        "are invisible to the same predicate purely because their names lack the substring -- so "
        "this entry records a naming coincidence, not a decision about geometry."
    ),
}


def _module_level_geometry_constants(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for node in tree.body:
        targets = (
            [node.target] if isinstance(node, ast.AnnAssign) else node.targets if isinstance(node, ast.Assign) else []
        )
        for t in targets:
            if isinstance(t, ast.Name) and any(k in t.id.upper() for k in _GEOMETRY_NAME):
                out.add(t.id)
    return out


def test_every_geometry_constant_is_declared_or_explicitly_exempt():
    from silly_kicks.tracking._feature_contract import DECLARED_CONSTANT_SOURCES

    declared = set(DECLARED_CONSTANT_SOURCES)
    undeclared = {}
    for path in _MODULES:
        for name in _module_level_geometry_constants(path):
            if name not in declared and name not in _EXEMPT:
                undeclared.setdefault(path.name, []).append(name)
    assert not undeclared, (
        f"undeclared geometry constants {undeclared}. Either declare the constant in the owning "
        f"model's feature contract (and extend contract_probe_frame so it is load-bearing), or "
        f"add it to _EXEMPT with a reason."
    )


def test_the_enumerator_is_not_vacuous():
    """A regex/AST gate that finds nothing passes silently forever."""
    found = set().union(*(_module_level_geometry_constants(p) for p in _MODULES))
    assert len(found) >= 4, f"enumerator found only {found}; it is not seeing the modules"


def test_no_dead_entries_in_either_list():
    """The other direction: a name that no longer exists anywhere is stale bookkeeping, and stale
    bookkeeping is how a list stops describing the thing it governs."""
    from silly_kicks.tracking._feature_contract import DECLARED_CONSTANT_SOURCES

    found = set().union(*(_module_level_geometry_constants(p) for p in _MODULES))
    stale_exempt = set(_EXEMPT) - found
    stale_declared = set(DECLARED_CONSTANT_SOURCES) - found
    assert not stale_exempt, f"_EXEMPT names constants that no longer exist: {sorted(stale_exempt)}"
    assert not stale_declared, (
        f"DECLARED_CONSTANT_SOURCES names constants that no longer exist: {sorted(stale_declared)}"
    )


def _built_contract_constants(tmp_path) -> dict[str, set[str]]:
    """The constants each model ACTUALLY stamps, read back from a real save()."""
    import json

    from silly_kicks.tracking._ghost_gk import GhostGkModel
    from silly_kicks.tracking._xcross_attempt import XCrossAttemptModel
    from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel

    out = {}
    for name, cls in (
        ("xshot", XShotOccurrenceModel),
        ("xcross", XCrossAttemptModel),
        ("ghost", GhostGkModel),
    ):
        d = tmp_path / name
        cls.from_variant("default").save(d)
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        out[name] = set(meta["feature_contract"]["constants"])
    return out


def test_the_registry_and_the_built_contracts_agree(tmp_path):
    """Close the loop BOTH ways.

    The registry is a name->key map; on its own it proves nothing about what save() stamps. Without
    this test a constant could be listed as "declared" while no contract carried it -- the gate
    would read as complete and enforce nothing, the exact failure mode a name-heuristic lint has.
    Reading the built artifacts is the only evidence that the declaration is real.
    """
    from silly_kicks.tracking._feature_contract import DECLARED_CONSTANT_SOURCES

    built = _built_contract_constants(tmp_path)
    all_keys = set().union(*built.values())
    registry_keys = set(DECLARED_CONSTANT_SOURCES.values())

    # (a) every key the registry claims is declared is stamped by at least one model
    assert registry_keys <= all_keys, (
        f"registry names keys no model stamps: {sorted(registry_keys - all_keys)}. Either a model "
        f"must declare it, or the source constant belongs in _EXEMPT."
    )
    # (b) no model stamps a key NOTHING accounts for. Accounted for = a module constant the registry
    # maps (the enumeration gate sees edits to it) OR a canonical `spadlconfig` source
    # (test_declared_constant_values.py pins its value). Requiring the registry alone made this
    # unsatisfiable the moment an extractor migrated -- see the migration test below.
    unaccounted = _unaccounted_keys(all_keys, DECLARED_CONSTANT_SOURCES)
    assert not unaccounted, (
        f"models stamp keys nothing accounts for: {sorted(unaccounted)}. EITHER add the owning "
        f"module constant to DECLARED_CONSTANT_SOURCES (if the extractor keeps a private copy), OR "
        f"add the key to CANONICAL_CONTRACT_KEYS (if it now reads spadlconfig directly)."
    )


def test_every_declared_constant_is_load_bearing_on_the_probe(tmp_path):
    """Pinned per-model, so a reviewer can see in one screen that xS declares no penalty-area
    constant (it has none) and that ghost declares the pair.

    Ghost's pair is now the CANONICAL one (ADR-050 §6 closed): its predicate and its declaration both
    read ``spadlconfig``, where it previously declared a 40.3-derived 20.15. This gate compares key
    NAMES only -- which is exactly how that divergence survived -- so the VALUES are pinned
    separately by ``tests/tracking/test_declared_constant_values.py``.

    A declaration the probe cannot move is a guard that fires when nothing changed -- which is how
    ``legacy_override`` becomes reflex.
    """
    built = _built_contract_constants(tmp_path)
    assert built["xshot"] == {"goal_width"}
    assert built["xcross"] == {"penalty_area_half_width", "penalty_area_depth", "goal_width"}
    assert built["ghost"] == {"penalty_area_half_width", "penalty_area_depth"}


# --------------------------------------------------------------------------------------------
# A key with a CANONICAL source needs no module constant -- migration is success, not a gap.


def _unaccounted_keys(all_keys: set[str], registry: dict[str, str]) -> set[str]:
    """Stamped keys traceable to NOTHING that gates them.

    A stamped key is accounted for by EITHER a module-level constant the registry maps (so the
    enumeration gate sees any edit to it) OR a canonical ``spadlconfig`` source (so
    ``test_declared_constant_values.py`` pins its VALUE). Requiring the registry alone makes the
    gate unsatisfiable the moment an extractor migrates: the constant is deleted BY DESIGN, and the
    old failure message tells the reader to re-add a constant that cannot exist.
    """
    from silly_kicks.tracking._feature_contract import CANONICAL_CONTRACT_KEYS

    return all_keys - (set(registry.values()) | set(CANONICAL_CONTRACT_KEYS))


def test_the_gate_survives_an_extractor_migrating_onto_the_canonical_source(tmp_path):
    """The post-xCross-migration state, simulated on the REAL stamped keys.

    ghost migrated in this cycle, so xCross's `_BOX_*` are the only module constants left mapping to
    `penalty_area_*`. Migrating xCross the same way -- the obvious next step -- deletes them, and
    prong (b) then fails with "Add the owning module constant", a remedy that is IMPOSSIBLE because
    the constant now lives in `spadlconfig`. The gate would block the very migration ADR-050 §6
    prescribes.
    """
    from silly_kicks.tracking._feature_contract import CANONICAL_CONTRACT_KEYS, DECLARED_CONSTANT_SOURCES

    all_keys = set().union(*_built_contract_constants(tmp_path).values())
    migrated = {n: k for n, k in DECLARED_CONSTANT_SOURCES.items() if k not in CANONICAL_CONTRACT_KEYS}
    assert set(migrated.values()) != set(DECLARED_CONSTANT_SOURCES.values()), (
        "the simulation removed nothing -- it is not exercising a migration"
    )
    assert _unaccounted_keys(all_keys, migrated) == set()


def test_a_key_with_neither_a_constant_nor_a_canonical_source_is_REPORTED():
    """Non-vacuity, and the property that keeps the union honest: widening the accounting must not
    make it accept everything. A stamped key backed by nothing must still fail."""
    assert _unaccounted_keys({"invented_key"}, {}) == {"invented_key"}


def test_every_canonical_contract_key_resolves_on_spadlconfig():
    """`CANONICAL_CONTRACT_KEYS` is resolved by `getattr(spadlconfig, key)`; a name that does not
    resolve would silently excuse a stamped key while pinning nothing."""
    import silly_kicks.spadl.config as spadlconfig
    from silly_kicks.tracking._feature_contract import CANONICAL_CONTRACT_KEYS

    assert CANONICAL_CONTRACT_KEYS, "an empty canonical set makes the accounting registry-only again"
    for key in CANONICAL_CONTRACT_KEYS:
        assert hasattr(spadlconfig, key), f"{key} does not resolve on spadlconfig"


def test_goal_width_is_NOT_canonical_and_that_is_recorded_not_accidental():
    """`spadlconfig` has no `goal_width` (verified 2026-08-14), so the goal-mouth keys are held up
    ONLY by their module constants -- the same un-migrated state ghost's pair was in before ADR-050
    §6. Pinning it here means a future `spadlconfig.goal_width` is a deliberate change to this gate,
    not a silent widening of what the accounting excuses."""
    import silly_kicks.spadl.config as spadlconfig
    from silly_kicks.tracking._feature_contract import CANONICAL_CONTRACT_KEYS

    assert "goal_width" not in CANONICAL_CONTRACT_KEYS
    assert not hasattr(spadlconfig, "goal_width")
