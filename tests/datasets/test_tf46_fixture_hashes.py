"""Offline integrity gate for the TF-46 Arm 1 slim fixtures (Task 2.1, spec section 10.1).

Each committed fixture's SHA-256 matches ``tests/datasets/tf46_fixture_sha256sums.txt``. No token, no
network: this is the default-CI guard behind the owner/DGX-run ``scripts/regen_tf46_fixtures.py``. The
manifest paths are relative to ``tests/datasets/``.
"""

import hashlib
from pathlib import Path

import pytest

_DATASETS = Path(__file__).parent
_MANIFEST = _DATASETS / "tf46_fixture_sha256sums.txt"


def _entries() -> list[tuple[str, str]]:
    rows = []
    for line in _MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        rows.append((rel, digest))
    return rows


def test_manifest_covers_both_providers():
    rels = [r for r, _ in _entries()]
    assert len(rels) == 54  # 20 skillcorner x 2 + 7 idsse x 2 + nothing else
    assert sum(r.startswith("skillcorner/opendata/") for r in rels) == 40
    assert sum(r.startswith("idsse/") for r in rels) == 14


@pytest.mark.parametrize("entry", _entries(), ids=lambda e: e[0])
def test_fixture_sha256_matches_manifest(entry):
    rel, digest = entry
    path = _DATASETS / rel
    assert path.exists(), f"missing fixture {rel}"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"sha256 drift in {rel}"
