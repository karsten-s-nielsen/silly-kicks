"""Corner bundled-weights gate (receiver pattern): RunZoneModel + CornerRoutineModel ship with full
provenance + a wheel model card, to the standard of the tracking peers.

The corner weights are params-only npz bundled IN the wheel (not HF-published; an xy-snapshot encoder
does not fit the frame-scorer Hub harness -- `validate_hub_variants`). This gate is corner-scoped: the
tracking POLICY / CARD / CONTRACT registries (`test_bundled_weights_corpus_policy`,
`test_model_cards_match_bundles`, `test_feature_contract`) only see `silly_kicks/tracking`, so the
corner bundles get this parallel gate instead. The repo-wide `test_bundled_weights_classification`
(float32 policy) and `test_artifact_provenance_output` DO glob all of `silly_kicks`, so the corner
dirs are registered there too. Complete by enumeration (ADR-056): the discovered bundle set equals the
registry exactly. Per bundle dir:

  - `training_commit` non-null (ADR-011 provenance; the repo-wide `test_artifact_provenance_output`
    also enforces this, pinned here corner-scoped for a close, legible failure);
  - only params-only + the model card ship (no raw provider data / reversible ids -- the reversibility
    policy that lets a derived artifact be redistributed);
  - a `MODEL_CARD.md` stating a provenance line that cites the bundle's `training_commit`;
  - the model loads fail-closed (RunZoneModel: SHA256 + feature-contract + chirality; CornerRoutineModel:
    SHA256 integrity).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_ROOT = Path("silly_kicks/corners")
#: bundle dir (relative to _ROOT) -> the public loader class on `silly_kicks.corners`.
_BUNDLES = {
    "_run_zone_weights/default": "RunZoneModel",
    "_routine_weights/default": "CornerRoutineModel",
}
#: The COMPLETE set of files a corner bundle may ship: learned parameters, metadata, the integrity
#: manifest, and the wheel model card. Anything else (raw frames, reversible ids) is a leak.
_ALLOWED_FILES = {"model.npz", "metadata.json", "SHA256SUMS", "MODEL_CARD.md"}


def _discover() -> set[str]:
    return {
        p.relative_to(_ROOT).as_posix()
        for base in _ROOT.glob("_*_weights")
        for p in base.glob("*")
        if p.is_dir() and p.name != "__pycache__" and not p.name.startswith(".")
    }


def _meta(rel: str) -> dict:
    return json.loads((_ROOT / rel / "metadata.json").read_text(encoding="utf-8"))


def test_population_is_exact():
    assert _discover() == set(_BUNDLES)


@pytest.mark.parametrize("rel", sorted(_BUNDLES))
def test_bundle_carries_training_commit(rel):
    assert _meta(rel).get("training_commit"), f"{rel} metadata has no training_commit (ADR-011 provenance)"


@pytest.mark.parametrize("rel", sorted(_BUNDLES))
def test_bundle_is_params_only(rel):
    extra = {p.name for p in (_ROOT / rel).iterdir() if p.is_file()} - _ALLOWED_FILES
    assert not extra, f"{rel} ships non-allowlisted files {sorted(extra)} -- bundles are params-only + card"


@pytest.mark.parametrize("rel", sorted(_BUNDLES))
def test_bundle_card_cites_its_training_commit(rel):
    card = (_ROOT / rel / "MODEL_CARD.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    commit = _meta(rel)["training_commit"]
    assert "**Provenance" in card, f"{rel} MODEL_CARD.md carries no provenance line"
    assert commit[:7] in card, f"{rel} MODEL_CARD.md must cite its training_commit {commit[:7]}"


@pytest.mark.parametrize("rel", sorted(_BUNDLES))
def test_bundle_loads_fail_closed(rel):
    import silly_kicks.corners as corners

    getattr(corners, _BUNDLES[rel]).load(_ROOT / rel)  # SHA256 + (RunZoneModel) feature-contract + chirality
