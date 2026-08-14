"""Every declared geometry constant must EQUAL the canonical one -- in the CODE and in the ARTIFACT.

ADR-050's enumeration gate asserts on constant NAMES, never VALUES
(``test_geometry_constant_enumeration.py`` compares key sets at ``:120`` and ``:158-160``). So an
extractor can migrate its PREDICATE to the canonical constant while its DECLARATION still derives
from a local one, and stamp an artifact that lies about the geometry it was fit on -- with every
existing gate green. That is exactly what ghost did before the ADR-050 §6 closure: it declared
**20.15** while the canonical value is **20.16**.

Keyed on the CANONICAL NAME rather than on where a constant happens to live, so it survives
constants being relocated out of extractor modules -- which is precisely what the ghost migration
does, and which would otherwise silently narrow the enumeration gate's coverage to xCross alone
(its ``test_the_enumerator_is_not_vacuous`` only asserts ``len(found) >= 4`` across ALL modules).
"""

from __future__ import annotations

import json
import pathlib

import pytest

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.tracking._feature_contract import CANONICAL_CONTRACT_KEYS

_WEIGHTS = {
    "ghost": "silly_kicks/tracking/_ghost_gk_weights/default/metadata.json",
    "xshot": "silly_kicks/tracking/_xshot_weights/default/metadata.json",
    "xcross": "silly_kicks/tracking/_xcross_weights/default/metadata.json",
}

#: Canonical source for each declared contract key this gate covers, SINGLE-SOURCED from the library
#: registry so the two cannot drift. `CANONICAL_CONTRACT_KEYS` is also what the enumeration gate's
#: accounting consults, so a key excused there is necessarily pinned here -- the property that makes
#: widening that accounting safe rather than a hole.
_CANONICAL = {key: (lambda k=key: getattr(spadlconfig, k)) for key in sorted(CANONICAL_CONTRACT_KEYS)}

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _stamped(model: str) -> dict:
    """What the shipped artifact declares."""
    meta = json.loads((_ROOT / _WEIGHTS[model]).read_text(encoding="utf-8"))
    return meta.get("feature_contract", {}).get("constants", {})


def _built(model: str) -> dict:
    """What the CODE declares right now -- distinct from what is stamped on disk."""
    from silly_kicks.tracking import _ghost_gk, _xcross_attempt, _xshot_occurrence

    builder = {
        "ghost": _ghost_gk._feature_contract_block,
        "xshot": _xshot_occurrence._feature_contract_block,
        "xcross": _xcross_attempt._feature_contract_block,
    }[model]
    return builder()["constants"]


@pytest.mark.parametrize("model", sorted(_WEIGHTS))
def test_declared_values_equal_the_canonical_values(model):
    """ARTIFACT-level: what ships must not lie about the geometry it was fit on."""
    for key, value in _stamped(model).items():
        if key not in _CANONICAL:
            continue  # goal_width etc. have their own canonical source
        assert value == _CANONICAL[key](), (
            f"{model} declares {key}={value} but the canonical value is {_CANONICAL[key]()}. "
            f"An artifact that declares a constant it was not fit on is exactly what ADR-050's "
            f"contract exists to prevent."
        )


@pytest.mark.parametrize("model", sorted(_WEIGHTS))
def test_built_values_equal_the_canonical_values(model):
    """CODE-level, and it carries a DIFFERENT meaning from its artifact-level sibling.

    ``_stamped()`` reads ``metadata.json``, which only refreshes at stamp time. Between a code
    migration and its re-stamp -- roughly an entire corpus pass -- there would otherwise be no
    signal that the declaration and the predicate now agree, and one red would conflate "the code is
    wrong" with "the artifact is stale". This one goes green the moment the code is fixed. It also
    fires in the change that INTRODUCES a future divergence rather than in the one that stamps it.
    """
    for key, value in _built(model).items():
        if key not in _CANONICAL:
            continue
        assert value == _CANONICAL[key](), (
            f"{model}'s live contract block declares {key}={value}, canonical is {_CANONICAL[key]()}"
        )


def test_the_check_is_not_vacuous():
    """At least one model must actually declare a key this gate covers, or it asserts nothing."""
    covered = [m for m in _WEIGHTS if set(_stamped(m)) & set(_CANONICAL)]
    assert covered, f"no model declares any of {sorted(_CANONICAL)}; this gate is inert"
