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
    # (b) no model stamps a key the registry does not know about
    assert all_keys <= registry_keys, (
        f"models stamp undeclared keys: {sorted(all_keys - registry_keys)}. Add the owning module "
        f"constant to DECLARED_CONSTANT_SOURCES so the enumeration gate can see it."
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
