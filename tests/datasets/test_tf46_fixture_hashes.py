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
    # 20 skillcorner x 2 + the 1886347 tracking bronze slim (R.2 2026-10-23) + 7 idsse x 2 + nothing else
    assert len(rels) == 55
    assert sum(r.startswith("skillcorner/opendata/") for r in rels) == 41
    assert [r for r in rels if r.endswith("_tracking_bronze_slim.parquet")] == [
        "skillcorner/opendata/1886347_tracking_bronze_slim.parquet"
    ]
    assert sum(r.startswith("idsse/") for r in rels) == 14


@pytest.mark.parametrize("entry", _entries(), ids=lambda e: e[0])
def test_fixture_sha256_matches_manifest(entry):
    rel, digest = entry
    path = _DATASETS / rel
    assert path.exists(), f"missing fixture {rel}"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"sha256 drift in {rel}"


def test_opendata_gi_frame_facts_present_b08():
    # TF-46 Arm 1 Part 2 Task 2.3 Step 3 (CL-1 evidence, B-08): every regenerated opendata fixture carries the GI
    # possession frame ids and attacking side, non-null on EVERY player_possession row (else OD-2).
    import pandas as pd

    files = sorted((_DATASETS / "skillcorner" / "opendata").glob("*_events_slim.csv"))
    assert len(files) == 20
    for f in files:
        ev = pd.read_csv(f, low_memory=False)
        pp = ev[ev["event_type"] == "player_possession"]
        assert len(pp) > 0, f.name
        for col in ("frame_start", "frame_end", "attacking_side"):
            assert col in ev.columns and pp[col].notna().all(), (f.name, col)
        assert set(pp["attacking_side"].unique()) <= {"left_to_right", "right_to_left"}, f.name
