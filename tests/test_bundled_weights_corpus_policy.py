"""G2 (combined-cycle-completion spec section 5): every bundled model variant dir carries the corpus
policy it must ship with. Complete by enumeration (ADR-056): the discovered dir set equals the registry
exactly; `_UNDERIVABLE` is asserted empty. Every expected value is copied from committed or archived
metadata -- a value here that no artifact carries is a defect in this test.
"""

import json
from pathlib import Path

import pytest

_ROOT = Path("silly_kicks/tracking")
_WHEEL_EXCLUDED = {"full"}  # pyproject.toml:222 excludes the maintainer-local `full` dirs from the wheel
_GHOST_PROVIDERS = ["gradientsports", "skillcorner", "sportec"]
_PUBLIC_PROVIDERS = ["idsse", "skillcorner"]


def _x_policy():
    return {("metadata.json", "shipped_variant"): "public", ("metadata.json", "provider_list"): _PUBLIC_PROVIDERS}


def _ghost_policy():
    return {
        ("metadata.json", "corpus_provenance.providers"): _GHOST_PROVIDERS,
        ("metadata.json", "corpus_provenance.n_games"): 179,
    }


def _gof_policy(variant):
    return {
        ("metadata.json", "corpus_provenance.n_games"): 179,
        ("metadata.json", "corpus_provenance.variant"): variant,
    }


POLICY: dict[str, dict[tuple[str, str], object]] = {
    "_xshot_weights/default": _x_policy(),
    "_xshot_weights/position_only": _x_policy(),
    "_xcross_weights/default": _x_policy(),
    "_xcross_weights/position_only": _x_policy(),
    "_ghost_gk_weights/default": _ghost_policy(),
    "_ghost_gk_weights/position_only": _ghost_policy(),
    "_ghost_gk_weights/sweeper": _ghost_policy(),
    "_ghost_gk_weights/sweeper_position_only": _ghost_policy(),
    "_ghost_outfield_weights/default": _gof_policy("default"),
    "_ghost_outfield_weights/position_only": _gof_policy("position_only"),
    "_gk_completion_weights/default": {
        ("metrics.json", "providers"): ["gradientsports"],
        ("metrics.json", "artifact_label"): "full",  # owner decision 2026-08-02 (test_gk_completion_taxonomy.py:48)
    },
    "_gk_completion_weights/skillcorner": {
        ("metrics.json", "variant"): "skillcorner",
        ("metrics.json", "n_matches"): 10,
    },
    "_receiver_weights/default": {
        ("metrics.json", "providers_trained"): ["statsbomb"],
        ("metrics.json", "corpus_visibility"): "public",  # as committed; C2 sets it per D7
    },
}

_UNDERIVABLE: tuple[str, ...] = ()


def _discover() -> set[str]:
    return {
        p.relative_to(_ROOT).as_posix()
        for p in _ROOT.glob("_*_weights/*")
        if p.is_dir() and p.name != "__pycache__" and not p.name.startswith(".") and p.name not in _WHEEL_EXCLUDED
    }


def _get(doc: dict, dotted: str):
    for part in dotted.split("."):
        doc = doc[part]
    return doc


def violations(dirname: str, policy: dict, read=None) -> list[str]:
    """The policy entries the dir's committed metadata does not satisfy (empty == compliant)."""
    read = read or (lambda f: json.loads((_ROOT / dirname / f).read_text(encoding="utf-8")))
    bad = []
    for (fname, key), want in policy.items():
        try:
            got = _get(read(fname), key)
        except (FileNotFoundError, KeyError) as exc:
            bad.append(f"{dirname}/{fname}:{key} missing ({exc!r})")
            continue
        if got != want:
            bad.append(f"{dirname}/{fname}:{key} = {got!r}, policy {want!r}")
    return bad


def test_every_variant_dir_is_registered_exactly():
    found, declared = _discover(), set(POLICY)
    assert found == declared, f"unregistered: {sorted(found - declared)}; missing: {sorted(declared - found)}"


def test_population_agrees_with_the_classification_registry():
    """Single-sourced population: every variant dir sits under a classified weights root."""
    from tests.test_bundled_weights_classification import WEIGHTS_CLASSIFICATION

    roots = {f"silly_kicks/tracking/{d.split('/')[0]}" for d in POLICY}
    assert roots <= set(WEIGHTS_CLASSIFICATION)


def test_underivable_is_empty():
    assert not _UNDERIVABLE


@pytest.mark.parametrize("dirname", sorted(POLICY))
def test_dir_matches_its_corpus_policy(dirname):
    assert not violations(dirname, POLICY[dirname])


@pytest.mark.parametrize("dirname", sorted(POLICY))
def test_recorded_runs_were_clean(dirname):
    p = _ROOT / dirname / "metrics.json"
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        if "run_tree_dirty" in m:
            assert m["run_tree_dirty"] is False, f"{dirname} was trained on a dirty tree"


def test_the_checker_catches_a_wrong_variant():
    """Anti-rot: an sc_extended artifact in a public slot must be reported."""
    fake = {"metadata.json": {"shipped_variant": "sc_extended", "provider_list": _PUBLIC_PROVIDERS}}
    assert violations("_xshot_weights/default", _x_policy(), read=fake.__getitem__)
