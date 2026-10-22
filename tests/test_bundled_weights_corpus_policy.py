"""G2 (combined-cycle-completion spec section 5): every bundled model variant dir carries the corpus
policy it must ship with. Complete by enumeration (ADR-056): the discovered dir set equals the registry
exactly; `_UNDERIVABLE` is asserted empty.

ADR-128 (spec 5.7, INV-9): the registry is the artifact manifest (`docs/artifacts/manifest.toml`) --
every expected value (variant, provider list, reproducibility, corpus rule, training commit and where it
is recorded) is declared there, copied from committed or archived metadata; a value there that no
artifact carries is a defect. The checker is `scripts.hf_jobs.manifest.violations`.
"""

import json
from pathlib import Path

import pytest

from scripts._corpus import bundled_public_arm_pairs
from scripts.hf_jobs import manifest as M
from scripts.hf_jobs.datasets import locked

_ROOT = Path("silly_kicks/tracking")
_WHEEL_EXCLUDED = {"full"}  # pyproject.toml:222 excludes the maintainer-local `full` dirs from the wheel
_MAN = M.load()
# The tracking wheel artifacts, keyed by their dir relative to silly_kicks/tracking.
POLICY: dict[str, M.Artifact] = {
    Path(a.path).relative_to(_ROOT).as_posix(): a
    for a in _MAN.artifacts.values()
    if a.kind == "wheel" and Path(a.path).is_relative_to(_ROOT)
}

_UNDERIVABLE: tuple[str, ...] = ()


def _discover() -> set[str]:
    return {
        p.relative_to(_ROOT).as_posix()
        for p in _ROOT.glob("_*_weights/*")
        if p.is_dir() and p.name != "__pycache__" and not p.name.startswith(".") and p.name not in _WHEEL_EXCLUDED
    }


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


def test_the_staged_public_ids_are_the_bundled_public_arm():
    """The public-arm corpus a public xShot/xCross bundle must carry is the dataset lock's ``public_ids``;
    that staging record equals the code's bundled public arm (the pre-ADR-128 policy value)."""
    for a in POLICY.values():
        if a.corpus == "public_ids":
            assert locked(a.dataset or "")["public_ids"] == bundled_public_arm_pairs(), a.id


@pytest.mark.parametrize("dirname", sorted(POLICY))
def test_dir_matches_its_corpus_policy(dirname):
    assert not M.violations(POLICY[dirname])


@pytest.mark.parametrize("dirname", sorted(POLICY))
def test_recorded_runs_were_clean(dirname):
    p = _ROOT / dirname / "metrics.json"
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        if "run_tree_dirty" in m:
            assert m["run_tree_dirty"] is False, f"{dirname} was trained on a dirty tree"


def test_the_checker_catches_a_wrong_variant():
    """Anti-rot: an sc_extended artifact in a public slot must be reported."""
    fake = {"metadata.json": {"shipped_variant": "sc_extended", "provider_list": ["idsse", "skillcorner"]}}
    assert M.violations(POLICY["_xshot_weights/default"], read=fake.__getitem__)


def test_anchor_covers_every_dir_exactly():
    # Anti-rot: every discovered dir declares its training commit and where it is recorded (ADR-056 idiom).
    assert {d for d, a in POLICY.items() if a.commit_field and a.training_commit} == _discover()


@pytest.mark.parametrize("dirname", sorted(POLICY))
def test_dir_traces_to_its_adr_anchor(dirname):
    a = POLICY[dirname]
    fname, _, key = (a.commit_field or "").partition(":")
    got = json.loads((_ROOT / dirname / fname).read_text(encoding="utf-8")).get(key)
    assert got == a.training_commit, f"{dirname}/{fname}:{key} = {got!r}, anchor {a.training_commit!r}"


def test_ghost_position_only_reproducibility_note_names_its_own_commit():
    m = json.loads((_ROOT / "_ghost_gk_weights/position_only/metrics.json").read_text(encoding="utf-8"))
    assert m["reproducibility"] == "restricted"
    assert m["training_commit"] in m["reproducibility_note"], (
        "the note must cite the commit it cannot be reproduced from"
    )
