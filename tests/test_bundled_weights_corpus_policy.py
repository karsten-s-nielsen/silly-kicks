"""G2 (combined-cycle-completion spec section 5): every bundled model variant dir carries the corpus
policy it must ship with. Complete by enumeration (ADR-056): the discovered dir set equals the registry
exactly; `_UNDERIVABLE` is asserted empty. Every expected value is copied from committed or archived
metadata -- a value here that no artifact carries is a defect in this test.
"""

import json
from pathlib import Path

import pytest

from scripts._corpus import bundled_public_arm_pairs

_ROOT = Path("silly_kicks/tracking")
# Re-fit at the combined-cycle C2 release commit; the 6 reused dirs keep the F1b anchor 3ca609f.
_M = "b62c1f24a7a9e3361ce416b402ed27da4a59b9e6"
_REUSED = "3ca609f8ae4003f411f9939dfde38fb320ff00fc"
_WHEEL_EXCLUDED = {"full"}  # pyproject.toml:222 excludes the maintainer-local `full` dirs from the wheel
_GHOST_PROVIDERS = ["gradientsports", "skillcorner", "sportec"]
_PUBLIC_PROVIDERS = ["idsse", "skillcorner"]


def _x_policy():
    return {
        ("metadata.json", "shipped_variant"): "public",
        ("metadata.json", "provider_list"): _PUBLIC_PROVIDERS,
        ("metrics.json", "shipped_variant"): "public",
        ("metrics.json", "reproducibility"): "public",
        ("metrics.json", "corpus_match_ids"): bundled_public_arm_pairs(),
    }


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
    "_ghost_gk_weights/position_only": {**_ghost_policy(), ("metrics.json", "reproducibility"): "restricted"},
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
        ("metrics.json", "artifact_label"): "public",
        ("metrics.json", "all_public"): True,
        ("metrics.json", "requested_match_ids"): bundled_public_arm_pairs(("skillcorner",)),
    },
    "_receiver_weights/default": {
        ("metrics.json", "providers_trained"): ["statsbomb"],
        ("metrics.json", "corpus_visibility"): "restricted",  # D7: licensed SB360; weights non-reversible
    },
}

# ADR anchor per dir: 7 re-fit at the C2 run commit, 6 reused at the F1b commit. training_commit lives
# in metadata.json (ghost/gof/xshot/xcross); run_commit in metrics.json (gkc/receiver).
_ANCHOR: dict[str, tuple[str, str, str]] = {
    "_xshot_weights/default": ("metadata.json", "training_commit", _M),
    "_xshot_weights/position_only": ("metadata.json", "training_commit", _M),
    "_xcross_weights/default": ("metadata.json", "training_commit", _M),
    "_xcross_weights/position_only": ("metadata.json", "training_commit", _M),
    "_ghost_gk_weights/position_only": ("metadata.json", "training_commit", _M),
    "_gk_completion_weights/skillcorner": ("metrics.json", "run_commit", _M),
    "_receiver_weights/default": ("metrics.json", "run_commit", _M),
    "_ghost_gk_weights/default": ("metadata.json", "training_commit", _REUSED),
    "_ghost_gk_weights/sweeper": ("metadata.json", "training_commit", _REUSED),
    "_ghost_gk_weights/sweeper_position_only": ("metadata.json", "training_commit", _REUSED),
    "_ghost_outfield_weights/default": ("metadata.json", "training_commit", _REUSED),
    "_ghost_outfield_weights/position_only": ("metadata.json", "training_commit", _REUSED),
    "_gk_completion_weights/default": ("metrics.json", "run_commit", _REUSED),
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


def test_anchor_covers_every_dir_exactly():
    # Anti-rot: the anchor registry's population equals the discovered dirs (ADR-056 idiom).
    assert set(_ANCHOR) == _discover()


@pytest.mark.parametrize("dirname", sorted(_ANCHOR))
def test_dir_traces_to_its_adr_anchor(dirname):
    fname, key, want = _ANCHOR[dirname]
    got = json.loads((_ROOT / dirname / fname).read_text(encoding="utf-8")).get(key)
    assert got == want, f"{dirname}/{fname}:{key} = {got!r}, anchor {want!r}"


def test_ghost_position_only_reproducibility_note_names_its_own_commit():
    m = json.loads((_ROOT / "_ghost_gk_weights/position_only/metrics.json").read_text(encoding="utf-8"))
    assert m["reproducibility"] == "restricted"
    assert m["training_commit"] in m["reproducibility_note"], (
        "the note must cite the commit it cannot be reproduced from"
    )
