"""A bundle's probe ids are only already-committed ids (combined-cycle spec 11.8; B r4 CCC-PLAN-40).

`probe_sample_matches` and the `probe_sample_in_training_folds` keys are written raw into the bundle's
metrics.json, which ships in the wheel and is copied into Hub publishes. Only the public arm and the two
GS probe ids already committed in the xcross `default` record (10502/10503) may appear."""

import json
from pathlib import Path

import pytest

from scripts._corpus import bundled_public_arm_pairs

_ALLOWED = {(str(p), str(m)) for p, m in bundled_public_arm_pairs()} | {
    ("gradientsports", "10502"),
    ("gradientsports", "10503"),
}
_DIRS = sorted(
    d
    for w in ("_xshot_weights", "_xcross_weights")
    for d in (Path("silly_kicks/tracking") / w).iterdir()
    if (d / "metrics.json").is_file()
)


def test_the_scan_covers_all_four_bundles():
    assert len(_DIRS) == 4  # never a silent pass over an empty glob


@pytest.mark.parametrize("d", _DIRS, ids=lambda d: f"{d.parent.name}/{d.name}")
def test_bundle_probe_ids_are_committed_ids(d):
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    pairs = {(str(p), str(mid)) for p, mid in (m.get("probe_sample_matches") or [])}
    assert pairs <= _ALLOWED, f"{len(pairs - _ALLOWED)} probe match(es) outside the committed set"
    assert {str(k) for k in (m.get("probe_sample_in_training_folds") or {})} <= {mid for _p, mid in _ALLOWED}
